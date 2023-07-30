#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_diy_model.py
@Author  :kaiwu
@Date    :2022/12/15 22:50 

'''


import numpy as np
import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
from conf.gorge_walk.config import DimConfig
from conf.gorge_walk.config import DIYConfig
from app.gorge_walk.sample_processor.gorge_walk_sample_processor import Frame
from app.gorge_walk.algorithm.torch_base_model import BaseModel


class DIYModel(BaseModel):
    """
        diy算法模型的实现: 包括神经网络、模型预测、模型训练、模型保存、模型恢复
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
        raise NotImplementedError

    def learn(self, g_data):
        """
            Description: 该方法实现了dqn算法和模型的训练过程
            ----------

            Return: 训练过程中产生的数据, 用于统计, 注意严格按照顺序监控面板上diy_1指标即该函数返回的第一个值, diy_2即该函数返回的第二个值, 以此类推, 总计是5个, 不需要统计监控的指标直接返回0即可
                    例如: 如果返回的是loss, clip_loss, v_loss, ent_loss, return, 则对应为diy_1, diy_2, diy_3, diy_4, diy_5
                          如果返回的是loss, clip_loss, 0, 0, 0, 则对应的diy_1, diy_2有监控数据, diy_3, diy_4, diy_5可忽略
            ----------

            Parameters
            ----------
            g_data: list
                由reverb传送过来的一个batch的原始训练数据

        """
        raise NotImplementedError
        
    
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
        raise NotImplementedError

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
        raise NotImplementedError