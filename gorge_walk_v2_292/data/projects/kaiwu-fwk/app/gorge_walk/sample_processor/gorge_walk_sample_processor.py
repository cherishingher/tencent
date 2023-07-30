#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
@Project :kaiwu-fwk 
@File    :gorge_walk_sample_processor.py
@Author  :kaiwu
@Date    :2022/11/15 20:57 

'''

import numpy as np
from framework.common.utils.singleton import Singleton
from conf.gorge_walk.config import AlgoConfig
from conf.gorge_walk.config import DimConfig
from framework.interface.sample_processor import SampleProcessor


class Frame(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


"""样本处理相关类"""

@Singleton
class GorgeWalkSampleProcessor(SampleProcessor):

    def __init__(self):

        self.must_need_sample_info = None

        self.agent_policy= []


    '''
    框架提供了日志接口, 业务直接使用即可
    '''

    def set_logger(self, logger):
        self.logger = logger

    '''
    sample manager init 处理
    '''

    def on_init(self, player_num, game_id):
        self.game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = player_num
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

        self.logger.info(
            f"sample sample on_init success, game_id {self.game_id}, num_agents {self.num_agents}")

    def should_train(self):
        return True

    def gen_expr(self, must_need_sample_info, network_sample_info):
        """
            Description: 该方法生成一个样本并插入self.m_replay_buffer, 
                根据算法需要提取must_need_sample_info和network_sample_info的信息
            ----------

            Return: None
            ----------

            Parameters
            ----------
            "must_need_sample_info": {
                "last_state": 当前帧的观测(包括legal action),
                "state": 下一帧的观测(包括legal action),
                "action": 动作,
                "info": {
                    "reward": 回报,
                    "done": 游戏结束flag,
                }
            }
            "network_sample_info": {
                "log_prob: 动作概率的对数,
                "value": 网络输出的value,
                "lstm_cell": lstm信息,
                "lstm_hidden": lstm信息,
            }

        """
        self.must_need_sample_info = must_need_sample_info

        for i in range(self.num_agents):
            last_state = must_need_sample_info['last_state'][i].get_state()['observation']
            state = must_need_sample_info['state'][i].get_state()['observation']
            last_state_legal_action = must_need_sample_info["last_state"][i].get_state()['legal_action']
            legal_action = must_need_sample_info["state"][i].get_state()['legal_action']
            action = must_need_sample_info['action'][i][1] * DimConfig.DIM_OF_ACTION_DIRECTION + must_need_sample_info['action'][i][0] 
            reward = must_need_sample_info['info'][i]['reward']
            done = must_need_sample_info['info'][i]['done']

            f = Frame(obs=last_state, _obs=state, act=action, rew=reward, done=1 if done else 0, ret=reward,\
                      obs_legal=last_state_legal_action, _obs_legal=legal_action)
            self.m_replay_buffer[i].append(f)


    def proc_exprs(self, del_last=False):
        """
        生成一个Episode的全量样本
        Returns: train_data_all
        """

        train_data_all = []
        for agent_data in self.m_replay_buffer:
            # agent_data: list[(frame_no,vec)]
            # obs, _obs, act, rew, ret, done
            tdata = agent_data 
            for frame in tdata:
                train_data_all.append({
                    # 发送样本时, 强制转换成float16
                    'input_datas': np.hstack((
                        np.array(frame.obs, dtype=np.float16),
                        np.array(frame._obs, dtype=np.float16),
                        np.array(frame.obs_legal, dtype=np.float16),
                        np.array(frame._obs_legal, dtype=np.float16),
                        np.array(frame.act, dtype=np.float16),
                        np.array(frame.rew, dtype=np.float16),
                        np.array(frame.ret, dtype=np.float16),
                        np.array(frame.done, dtype=np.float16),
                    )),
                })
        self.logger.debug(f'generate one Episode sampls')

        train_frame_cnt = len(train_data_all)
        drop_frame_cnt = 0
        self.reset(num_agents=self.num_agents, game_id=self.game_id)
        return train_data_all, train_frame_cnt, drop_frame_cnt
    
    def finalize(self):
        pass

    def reset(self, num_agents, game_id):
        self.game_id = game_id
        self.num_agents = num_agents
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]


@Singleton
class GorgeWalkSampleProcessorPPO(SampleProcessor):

    def __init__(self):
        self.must_need_sample_info = None
        self.network_sample_info = None

        self.agent_policy= []

        self.on_init(player_num=1, game_id=0)
    
    '''
    框架提供了日志接口, 业务直接使用即可
    '''
    def set_logger(self, logger):
        self.logger = logger

    '''
    sample manager init 处理
    '''
    def on_init(self, player_num, game_id):
        self.game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = player_num
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def should_train(self):
        return True

    def gen_expr(self, must_need_sample_info, network_sample_info):
        """
        生成一个样本
        """
        self.must_need_sample_info = must_need_sample_info

        for i in range(self.num_agents):
            last_state = must_need_sample_info['last_state'][i].get_state()['observation']
            legal_action = must_need_sample_info["last_state"][i].get_state()['legal_action']
            state = must_need_sample_info['state'][i].get_state()['observation']
            action = must_need_sample_info['action'][i][1] * DimConfig.DIM_OF_ACTION_DIRECTION + must_need_sample_info['action'][i][0] 
            reward = must_need_sample_info['info'][i]['reward']
            done = must_need_sample_info['info'][i]['done']
            
            log_prob = network_sample_info['log_prob']
            value = network_sample_info['value']

            f = Frame(obs=last_state, legal_action=legal_action, act=action, rew=reward, done=1 if done else 0, ret=reward, adv=reward, log_prob=log_prob, value=value, _value=0.)

            if len(self.m_replay_buffer[i]) > 0:
                self.m_replay_buffer[i][-1]._value = value
            
            self.m_replay_buffer[i].append(f)


    def proc_exprs(self, del_last=False):
        """
        生成一个Episode的全量样本
        Returns: train_data_all
        """

        train_data_all = []
        for agent_data in self.m_replay_buffer:
            # agent_data:list[(frame_no,vec)]
            # obs, _obs, act, rew, ret, done, adv, log_prob
            tdata = self.__gdata2tdata(agent_data)
            for frame in tdata:
                train_data_all.append({
                    # 发送样本时, 强制转换成float32
                    'input_datas': np.hstack((
                        np.array(frame.obs, dtype=np.float32),
                        np.array(frame.legal_action, dtype=np.float32),
                        np.array(frame.act, dtype=np.float32),
                        np.array(frame.ret, dtype=np.float32),
                        np.array(frame.adv, dtype=np.float32),
                        np.array(frame.log_prob, dtype=np.float32),
                    )),
                })
            
        train_frame_cnt = len(train_data_all)
        drop_frame_cnt = 0
        self.reset(num_agents=self.num_agents, game_id=self.game_id)
        return train_data_all, train_frame_cnt, drop_frame_cnt
    
    def finalize(self):
        pass
    
    def __gdata2tdata(self, g_data):
        gae, last_gae = 0., 0.
        _gamma = AlgoConfig.GAMMA
        _lamda = AlgoConfig.LAMDA
        rew = [frame.rew for frame in g_data]
        value = [frame.value for frame in g_data]
        next_value = [frame._value for frame in g_data]
        for i in range(len(g_data) - 1, -1, -1):
            if not g_data[i].done:
                delta = rew[i] + _gamma * next_value[i] - value[i]
            else:
                delta = rew[i] - value[i]
            gae = gae * _gamma * _lamda + delta
            g_data[i].adv = gae
            g_data[i].ret = gae + value[i]
        
        return g_data

    def reset(self, num_agents, game_id):
        self.game_id = game_id
        self.num_agents = num_agents
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]


@Singleton
class GorgeWalkSampleProcessorDIY(SampleProcessor):

    def __init__(self):
        self.must_need_sample_info = None

        self.agent_policy= []

        self.on_init(player_num=1, game_id=0)
    
    '''
    框架提供了日志接口, 业务直接使用即可
    '''
    def set_logger(self, logger):
        self.logger = logger

    '''
    sample manager init 处理
    '''
    def on_init(self, player_num, game_id):
        self.game_id = game_id
        self.m_task_id, self.m_task_uuid = 0, "default_task_uuid"
        self.num_agents = player_num
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def should_train(self):
        return True

    def gen_expr(self, must_need_sample_info, network_sample_info):
        """
        生成一个样本
        """
        raise NotImplementedError


    def proc_exprs(self, del_last=False):
        """
        生成一个Episode的全量样本
        Returns: train_data_all
        """
        raise NotImplementedError
    
    def finalize(self):
        pass

    def reset(self, num_agents, game_id):
        self.game_id = game_id
        self.num_agents = num_agents
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]
