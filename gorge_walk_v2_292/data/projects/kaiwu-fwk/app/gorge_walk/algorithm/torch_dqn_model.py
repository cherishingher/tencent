#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_dqn_model.py
@Author  :kaiwu
@Date    :2023/7/15 22:50 

'''


import numpy as np
import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
from conf.gorge_walk.config import DimConfig, DQNConfig
from app.gorge_walk.sample_processor.gorge_walk_sample_processor import Frame
from copy import deepcopy
from app.gorge_walk.algorithm.torch_base_model import BaseModel


class DQNModel(BaseModel):
    """
        dqn算法模型的实现: 包括神经网络、模型预测、模型训练、模型保存、模型恢复
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
        self.model = network
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=DQNConfig.START_LR)
        self._eps = np.finfo(np.float32).eps.item()
        self._gamma = DQNConfig.GAMMA
        self.name = name
        self.num_head = 2

        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.file_queue = []
        self.predict_count = 0

    def epsilon(self):
        return min(0.9, 0.5 + self.predict_count / DQNConfig.EPSILON_GREEDY_PROBABILITY)

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, g_data):
        """
            Description: 该方法实现了dqn算法和模型的训练过程
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
        obs_legal = torch.tensor(np.array([frame.obs_legal for frame in t_data]))
        obs_legal = torch.cat((obs_legal[:,0].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_ACTION_DIRECTION), \
                                obs_legal[:,1].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_TALENT)), 1).bool().to(self.model.device)
        _obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _obs_legal = torch.cat((_obs_legal[:,0].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_ACTION_DIRECTION), \
                                _obs_legal[:,1].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_TALENT)), 1).bool().to(self.model.device)

        action = torch.LongTensor([frame.act for frame in t_data]).view(-1,1).long().to(self.model.device)
        ret = torch.tensor(np.array([frame.ret for frame in t_data]), device=self.model.device)
        _obs = [frame._obs for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.model.device)

        model = getattr(self, 'model')
        model.eval()
        with torch.no_grad():
            q, h = model(_obs, state=None)
            q = q.masked_fill(~_obs_legal, float('-inf'))    
            q_max = q.max(dim=1).values.detach()#.cpu()
        
        target_q = ret + self._gamma * q_max * not_done

        self.optim.zero_grad()
        frames = self(obs, model_selector="model", model_mode="train")
        loss = torch.square(target_q - frames.logits.gather(1, action).view(-1)).sum() 
        loss.backward()
        self.optim.step()

        self.train_step += 1
        # 更新target网络
        if self.train_step % DQNConfig.TARGET_UPDATE_FREQ == 0:
            self.update_target_q()
            
        # 返回统计数据
        loss_value = loss.detach().item()
        return loss_value, loss_value, target_q.mean().detach().item(), ret.mean().detach().item()
    
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
        legal_action = torch.cat((legal_action[:,0].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_ACTION_DIRECTION),\
            legal_action[:,1].unsqueeze(1).expand(len(obs), DimConfig.DIM_OF_TALENT)), 1).bool().to(self.model.device)

        with torch.no_grad():
            if types == "max":
                logits, _ = model(obs, state=state)
                logits = logits.masked_fill(~legal_action, float('-inf'))
                act = logits.argmax(dim=1).cpu().view(-1,1).tolist()

            elif types == "prob":
                if np.random.rand(1) >= self.epsilon():    # epsilon greedy
                    random_action = np.random.rand(len(obs), DimConfig.DIM_OF_ACTION_DIRECTION + DimConfig.DIM_OF_TALENT)
                    random_action = torch.from_numpy(random_action).to(self.model.device)
                    random_action = random_action.masked_fill(~legal_action, 0)  
                    act = random_action.argmax(dim=1).cpu().view(-1,1).tolist()

                else:
                    logits, _ = model(obs, state=state)
                    logits = logits.masked_fill(~legal_action, float('-inf'))              
                    act = logits.argmax(dim=1).cpu().view(-1,1).tolist()

            else:
                raise AssertionError

        direction_space = DimConfig.DIM_OF_ACTION_DIRECTION
        format_action = [[instance[0]%direction_space, instance[0]//direction_space] for instance in act]
        network_sample_info = [(None, None)] * len(format_action)
        lstm_info = [(None, None)] * len(format_action)

        self.predict_count += 1

        return format_action, network_sample_info, lstm_info

    def __call__(self, obs, state=None, model_selector="model", model_mode="train"):
        model = getattr(self, model_selector)   
        getattr(model, model_mode)()            # model.train() or model.eval()
        logits, h = model(obs, state=state)
        return Frame(logits=logits)

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
                      _obs=i[DimConfig.observation_shape:2*DimConfig.observation_shape],
                      obs_legal=i[-8:-6], _obs_legal=i[-6:-4],
                      act=i[-4], rew=i[-3], ret=i[-2], done=i[-1]
            )for i in t_data]
    
    def load_param(self, path='/tmp/pyt-model', id='1'):
        self.model.load_state_dict(torch.load(f"{str(path)}/model.ckpt-{str(id)}.pkl",
                                   map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        self.update_target_q()
