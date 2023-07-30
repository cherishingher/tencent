#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_network.py
@Author  :kaiwu
@Date    :2022/11/11 12:47 

'''


import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
import numpy as np
from torch import nn
from conf.gorge_walk.config import Config, DimConfig


class BaseNetwork(nn.Module):
    """
        由pytorch实现的基础神经网络
        可以根据配置multi_head 选择是否多头输出, 如果是True, 则最后一层由具体网络定义
        可以根据配置use_softmax 选择是否在最后一层输出softmax
    """
    def __init__(self, state_shape, action_shape=(4, ), multi_head=False, use_softmax=False):
        super().__init__()
        state_shape = state_shape["observation"].shape
        action_shape = action_shape["a"].shape
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [
            nn.Linear(np.prod(state_shape), 256),
            nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(256, 256), nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        if not multi_head:
            self.layers += [nn.Linear(128, np.prod(action_shape))]
        if use_softmax:
            self.layers += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.layers).to(self.device)

    # 前向推理
    def forward(self, s, state=None, info={}):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float32)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, state

    # 加载最新模型文件
    def load_last_new_model(self, models_path):
        self.load_state_dict(torch.load(models_path))

class BaseCNNNetwork(nn.Module):
    """
        由pytorch实现的基础神经网络
        可以根据配置multi_head 选择是否多头输出, 如果是True, 则最后一层由具体网络定义
        可以根据配置use_softmax 选择是否在最后一层输出softmax
    """
    def __init__(self, state_shape, action_shape=(4, ), multi_head=False, use_softmax=False):
        super().__init__()
        state_shape = DimConfig.observation_shape_after_cnn
        action_shape = action_shape["a"].shape
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        cnn_layer1 = [
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()  
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ]
        max_pool = [
            nn.MaxPool2d(kernel_size=(2,2))
        ]
        self.cnn_layer = cnn_layer1 + max_pool + cnn_layer2 + max_pool + cnn_layer3 + max_pool
        self.cnn_model = nn.Sequential(*self.cnn_layer).to(self.device)

        fc_layer1 = [
            nn.Linear(np.prod(state_shape), 256),
            nn.ReLU(inplace=True)
        ]
        fc_layer2 = [
            nn.Linear(256, 128), 
            nn.ReLU(inplace=True)
        ]
        fc_layer3 = [nn.Linear(128, np.prod(action_shape))]
        
        self.fc_layers = fc_layer1 + fc_layer2
        
        if not multi_head:
            self.fc_layers += fc_layer3
        if use_softmax:
            self.fc_layers += [nn.Softmax(dim=-1)]
        
        self.model = nn.Sequential(*self.fc_layers).to(self.device)

    # 前向推理
    def forward(self, s, state=None, info={}):
        feature_vec, obstacle_map, end_map, memory_map = self.__split_ndarray(s, Config.DESC_OBS_SPLIT)

        feature_maps = torch.stack(
            [obstacle_map, end_map, memory_map], dim=1).to(self.device) 
        
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)                 

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)           
        
        logits = self.model(concat_feature)
        return logits, state

    # 加载最新模型文件
    def load_last_new_model(self, models_path):
        self.load_state_dict(torch.load(models_path))

    def __split_ndarray(self, array, list_desc):
        """
            Description: 将序列切分成规定格式的子序列
            Parameters
            ----------
            array : list 或者 np.ndarray
                原始序列
            list_desc : 
                描述子序列的数据结构
        """
        if isinstance(array, list):
            array = np.array(array)
        res = list()
        last_idx = 0
        for i in list_desc:
            if isinstance(i, tuple):
                current_idx = last_idx + (i[0] * i[1])
                s = torch.tensor(
                    array[:, last_idx:current_idx], device=self.device, dtype=torch.float32)
                batch = s.shape[0]
                s = s.view(batch, i[0], i[1])
            elif isinstance(i, int):
                current_idx = last_idx + i
                s = torch.tensor(
                    array[:, last_idx:current_idx], device=self.device, dtype=torch.float32)
                batch = s.shape[0]
                s = s.view(batch, -1)
            else:
                raise TypeError

            res.append(s)
            last_idx = current_idx
        return res


class ActorCriticNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.preprocess = BaseCNNNetwork(state_shape, action_shape, multi_head=True, use_softmax=False)
        self.policy_layer = nn.Linear(128, np.prod(action_shape["a"].shape)).to(self.device)
        self.value_layer = nn.Linear(128, 1).to(self.device)

    def forward(self, s, state=None, info={}):
        _mid, h = self.preprocess(s, state)
        logits = self.policy_layer(_mid)
        value = self.value_layer(_mid)
        return logits, value

    def load_last_new_model(self, models_path):
        self.load_state_dict(torch.load(models_path))

