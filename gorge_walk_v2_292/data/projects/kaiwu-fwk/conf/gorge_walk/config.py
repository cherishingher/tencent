#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :config.py
@Author  :kaiwu
@Date    :2023/7/1 10:37 

'''

# 关于维度的配置
class DimConfig:

    observation_shape = 8072                # observation的大小
    legal_action_shape = 2                  # legal-action的维度
    observation_shape_after_cnn = 4365      # 经过cnn之后的特征维度

    DIM_OF_ACTION_DIRECTION = 8             # 移动动作方向的维度
    DIM_OF_TALENT = 8                       # 闪现动作方向的维度

    sub_action_mask_shape = 0               # 以下是可以忽略的配置
    lstm_hidden_shape = 0
    lstm_cell_shape = 0

# 关于算法的参数配置
class AlgoConfig:
    GAMMA = 0.9                             # RL中的回报折扣GAMMA
    LAMDA = 0.95                            # 某些RL算法需要的折扣因子LAMDA
    
    START_LR = 5e-4                         # 初始的学习率 

# 关于框架使用的配置 
class Config:
    # pos_float + pos_onehot + organ + cd&talent, obstacle_map, treasure_map, end_map, location_memory
    #         2 +   128*2    +  9*1 +     2,     51*51*3
    DESC_OBS_SPLIT = [269, (51, 51), (51, 51), (51, 51)]   # sum = 8072

    MAX_FILE_KEEP_CNT = 300                 # 容器里保留的最大model文件数量 

# 关于DQN使用的配置
class DQNConfig(AlgoConfig):

    TARGET_UPDATE_FREQ = 1000               # target网络的更新频率     
    
    '''
    def epsilon(self):
        return min( 0.9, 0.5 + self.predict_count / DQNConfig.EPSILON_GREEDY_PROBABILITY)
    '''
    EPSILON_GREEDY_PROBABILITY = 30000      # 探索因子, epsilon的计算见上面注释中的函数

# 关于PPO使用的配置
class PPOConfig(AlgoConfig):

    EPS_CLIP = 0.1                          # PPO 中控制clip的参数
    W_VF = 0.5                              # value loss 的权重
    W_ENT = 0.05                            # entropy 的权重
    GRAD_NORM = None                        # 梯度进行Norm操作的参数, 默认None不进行Norm
    LSM_CONST_W = 1e10                      # 定义LSM常量W, W用于控制LSM的输出范围，越大输出范围越小
    LSM_CONST_E = 1e-5                      # 定义LSM常量E,  E用于控制LSM的输出精度，越小输出精度越高


# 关于DIY算法使用的配置
class DIYConfig(AlgoConfig):
    # 这部分需要参赛选手按需实现
    pass
