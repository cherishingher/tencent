#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :gorge_walk_config.py
@Author  :kaiwu
@Date    :2022/12/1 10:37 

'''

import json
from framework.common.utils.singleton import Singleton

# 读取配置文件
with open("/data/projects/kaiwu-fwk/conf/system/gorge_walk_config.json", 'r') as f:
    GW_CONFIG = json.load(f)
    
with open("/data/projects/kaiwu-fwk/conf/gorge_walk/gorge_walk_train_config.json", 'r') as f:
    GW_TRAIN_CONFIG = json.load(f)

with open("/data/projects/kaiwu-fwk/conf/system/map_data/position_pool.json", 'r') as f:
    POS_POOL = json.load(f)


@Singleton
class GWConfigControl(object):
    '''
    全局配置类
    '''
    def __init__(self) -> None:
        self.map_id = int(GW_CONFIG['nature_client']['map_id'][0])
        self.treasure_num = GW_TRAIN_CONFIG["nature_client"]["treasure_num"]
        self.map_name = f'map_{self.map_id}'

    def reset(self):
        '''
        game_id发生变化时更新map_name
        '''
        self.map_name = f'map_{self.map_id}'


GW2_CONFIG = GWConfigControl()