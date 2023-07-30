#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_ppo_model.py
@Author  :kaiwu
@Date    :2023/1/9 17:23 

'''

import numpy as np
import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
from torch import nn
import torch.nn.functional as F
from app.gorge_walk.algorithm.torch_base_model import BaseModel
from conf.gorge_walk.config import PPOConfig, DimConfig
from app.gorge_walk.sample_processor.gorge_walk_sample_processor import Frame


class PPOModel(BaseModel):
    """
        ppo算法模型的实现: 包括神经网络、模型预测、模型训练、模型保存、模型恢复
    """

    def __init__(self, network, name, role='actor'):
        """
            Parameters
            ----------
            network : torch_network.BaseNetwork
                神经网络通过参数传入
            name : str
                该模型的名字，用于标识
            role : str
                适配框架, 用于区分当前模型的使用场景(actor或learner), 当前模型不进行区分
        """
        super().__init__(network, name, role)
        self.name = name
        self.model =  network
        self.optim = torch.optim.Adam(self.model.parameters(), lr=PPOConfig.START_LR) 
        self.dist_fn = torch.distributions.Categorical
        self._eps = np.finfo(np.float32).eps.item()
        self._grad_norm = PPOConfig.GRAD_NORM
        self._eps_clip = PPOConfig.EPS_CLIP
        self._w_vf = PPOConfig.W_VF
        self._w_ent = PPOConfig.W_ENT
        self._lsm_const_w = PPOConfig.LSM_CONST_W
        self._lsm_const_e = PPOConfig.LSM_CONST_E

    def learn(self, g_data):
        """
            Description: 该方法实现了算法和模型的训练过程
            ----------

            Return: 训练过程中产生的数据, 用于统计
            ----------
            
            Parameters
            ----------
            g_data: list
                由reverb传送过来的一个batch的原始训练数据

        """

        # 将reverb传入的数据转换成可以训练的数据       
        t_data = self.__rdata2tdata(g_data)

        # 提取训练需要用到的数据
        obs = [frame.obs for frame in t_data]
        legal_action = torch.tensor(np.array([frame.legal_action for frame in t_data]), device=self.model.preprocess.device)
        adv = torch.tensor(np.array([frame.adv for frame in t_data]), device=self.model.preprocess.device)
        ret = torch.tensor(np.array([frame.ret for frame in t_data]), device=self.model.preprocess.device)
        log_prob_old = torch.tensor(np.array([frame.log_prob for frame in t_data]), device=self.model.preprocess.device)

        act = torch.tensor(np.array([frame.act for frame in t_data]), device=self.model.preprocess.device)
        
        adv = adv.type(torch.FloatTensor).to(self.model.preprocess.device)
        ret = ret.type(torch.FloatTensor).to(self.model.preprocess.device)
        log_prob_old = log_prob_old.type(torch.FloatTensor).to(self.model.preprocess.device)

        self.optim.zero_grad()

        tmp = self(obs, legal_action, model_selector='model', model_mode="train")
        dist, v = tmp.dist, tmp.state.flatten()
        
        log_prob = dist.log_prob(act).to(self.model.preprocess.device)

        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1. - self._eps_clip, 1. + self._eps_clip) * adv
        clip_loss = -torch.min(surr1, surr2).mean()

        v_loss = F.mse_loss(ret, v)
        ent_loss = dist.entropy().mean()

        loss = clip_loss + self._w_vf * v_loss - self._w_ent * ent_loss
        loss.backward()

        if self._grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()),
                max_norm=self._grad_norm,
            )

        self.optim.step()

        return loss.detach().item(), clip_loss.detach().item(), v_loss.detach().item(), ent_loss.detach().item(), ret.mean().detach().item()
    
    def get_action(self, *kargs, **kwargs):
        return self.predict(*kargs, **kwargs)

    def predict(self, obs, state=None, types="prob", model_selector="model"):
        """
            Description: 该方法实现了模型的预测
            ----------

            Return: 
            ----------
            format_action: list
                预测得到的动作序列
            network_sample_info: list
                返回的其他信息，该算法无需返回有效信息
            lstm_info: list
                返回的lstm相关信息, 该网络没有使用lstm, 则返回None

            Parameters:
            ----------
            obs: dict
                由aisvr传送过来的一个observation数据

        """
        model = getattr(self, model_selector)
        model.eval()
        legal_action = torch.tensor(np.array(obs["legal_action"]))
        obs = obs["observation"]
        with torch.no_grad():
            logits, value = model(obs, state=state)
            logits = self._legal_soft_max(logits, legal_action)
            if types == "max":
                dist = self.__logits2dist(logits)
                act = logits.argmax(dim=1)
                log_prob = dist.log_prob(act)
            elif types == "prob":
                dist = self.__logits2dist(logits)
                act = dist.sample()
                log_prob = dist.log_prob(act)
            else:
                raise AssertionError

        format_action = act.detach().cpu().numpy().tolist()
        log_prob = log_prob.detach().cpu().numpy().tolist()
        value = value.reshape(-1).detach().cpu().numpy().tolist()

        direction_space = DimConfig.DIM_OF_ACTION_DIRECTION
        format_action = [[instance%direction_space, instance//direction_space] for instance in format_action]
        network_sample_info = [(log_prob[i], value[i]) for i in range(len(format_action))]
        lstm_info = [(None, None)] * len(format_action)
        return format_action, network_sample_info, lstm_info
    
    def __call__(self, obs, legal_action, state=None, model_selector="model", model_mode="train"):
        model = getattr(self, model_selector)   
        getattr(model, model_mode)()            # model.train() or model.eval()
        logits, h = model(obs, state=state)
        logits = self._legal_soft_max(logits, legal_action)
        dist = self.__logits2dist(logits)
        act = dist.sample()
        return Frame(logits=logits, act=act, state=h, dist=dist)
    
    def _legal_soft_max(self, input_hidden, legal_action):
        """
            在过滤ilegal动作之后再softmax
        """
        legal_action = torch.cat((legal_action[:,0].unsqueeze(1).expand(legal_action.shape[0], DimConfig.DIM_OF_ACTION_DIRECTION),\
            legal_action[:,1].unsqueeze(1).expand(legal_action.shape[0], DimConfig.DIM_OF_TALENT)), 1).bool().to(self.model.device)

        tmp = input_hidden - self._lsm_const_w * (~legal_action)
        tmp_max = torch.max(tmp, 1, keepdims=True).values
        # Not necessary max clip 1
        tmp = torch.clamp(tmp - tmp_max, -self._lsm_const_w, 1)
        tmp = (torch.exp(tmp) + self._lsm_const_e) * legal_action
        probs = tmp / torch.sum(tmp, 1, keepdims=True)
        return probs
    
    def __rdata2tdata(self, r_data):
        """
            Description: 该方法将reverb传入的数据转换成可以训练的数据
            ----------

            Return: 
            ----------
            t_data: list
                训练数据

            Parameters
            ----------
            r_data: list
                由reverb传入的原始数据
        """

        t_data = list(r_data)
        return [ Frame(obs=i[:DimConfig.observation_shape], 
                       legal_action=i[-6:-4], act=i[-4], ret=i[-3], adv=i[-2], log_prob=i[-1]
            )for i in t_data]
    
    def __logits2dist(self, logits):
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        return dist

