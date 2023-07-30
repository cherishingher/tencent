#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :gorge_walk_action.py
@Author  :kaiwu
@Date    :2022/11/15 20:57 

'''

import numpy as np
from framework.interface.array_spec import ArraySpec
from framework.common.algorithms.distribution import CategoricalDist
from framework.interface.action import Action, ActionSpec
from conf.gorge_walk.config import DimConfig


class GorgeWalkAction(Action):
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {'a': self.a}

    @staticmethod
    def action_space():
        direction_space = DimConfig.DIM_OF_ACTION_DIRECTION
        talent_space = DimConfig.DIM_OF_TALENT
        return {'a': ActionSpec(ArraySpec((direction_space + talent_space), np.int32), pdclass=CategoricalDist)}

    def __str__(self):
        return str(self.a)