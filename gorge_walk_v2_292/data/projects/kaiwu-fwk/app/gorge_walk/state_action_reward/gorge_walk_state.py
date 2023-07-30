#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project: kaiwu-fwk 
@File    :gorge_walk_state.py
@Author  :kaiwu
@Date    :2022/10/20 11:43 

'''

import numpy as np
from framework.interface.array_spec import ArraySpec
from framework.interface.state import State
from conf.gorge_walk.config import DimConfig

'''
主要用于actor上使用
'''
class GorgeWalkState(State):
    def __init__(self, value):
        """
        Args:
            value: 由run_handler构造本类, 为on_update函数的一个返回值(当需要预测时)
        """
        self.value = value

    def get_state(self):
        """
        根据构造函数中传入的value,构造返回一个dict
        dict会传给Actor进行预测
        """
        observation = np.array(self.value["observation"], dtype=np.float64)
        legal_action = np.array(self.value['legal_action'], dtype=np.float64)
        sub_action_mask = np.array(
            self.value['sub_action_mask'], dtype=np.float64)
        lstm_hidden = np.array(self.value['lstm_hidden'], dtype=np.float64)
        lstm_cell = np.array(self.value['lstm_cell'], dtype=np.float64)
        return {
            'observation': observation,
            'legal_action': legal_action,
            'sub_action_mask': sub_action_mask,
            'lstm_hidden': lstm_hidden,
            'lstm_cell': lstm_cell
        }

    @staticmethod
    def state_space():
        """
        规定state中每个变量的shape, 必须为numpy数组
        Returns:
        """
        observation_shape = (DimConfig.observation_shape,)
        legal_action_shape = (DimConfig.legal_action_shape,)
        sub_action_mask_shape = (DimConfig.sub_action_mask_shape,)
        lstm_hidden_shape = (DimConfig.lstm_hidden_shape,)
        lstm_cell_shape = (DimConfig.lstm_cell_shape,)
        return {
            'observation':  ArraySpec(observation_shape, np.float64),
            'legal_action': ArraySpec(legal_action_shape, np.float64),
            'sub_action_mask': ArraySpec(sub_action_mask_shape, np.float64),
            'lstm_hidden': ArraySpec(lstm_hidden_shape, np.float64),
            'lstm_cell': ArraySpec(lstm_cell_shape, np.float64)
        }

    def __str__(self):
        return str(self.value)
