#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :gorge_walk_obs_reward.py
@Author  :kaiwu
@Date    :2023/1/9 17:23 

'''

import numpy as np
import json
from app.gorge_walk.environment.feature_process.feature_process import convert_pos_to_grid_pos
from app.gorge_walk.environment.protocl.common_pb2 import GameStatus, RelativeDirection
from framework.common.config.config_control import CONFIG
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine


REWARD_CONFIG = None


def parse_from_proto_to_state(req_pb):
    """
    该函数是特征处理的重要函数, 主要负责：
        - 从proto数据中解析原始数据
        - 从proto数据中解析特征
        - 对特征进行处理, 并返回处理后的特征向量
        - 特征的拼接
        - 合法动作的标注
    函数的输入：
        - req_pb: battlesrv发送的proto数据
    函数的输出：
        - observation: 特征向量
        - legal_action: 合法动作的标注
        - [[]]: 该返回值暂时不用
    注意: 
        - 该函数的返回值必须是 observation, legal_action, [[]] 这三个值, 第三个返回暂时不用
        - 默认的特征拼接里并没有加入宝箱相关的特征, 这部分需要同学们自行添加
    """

    observation, legal_action = [], []

    # 原始数据的解包，这里只提供了FrameState里的解包，如果需要更细节的数据，请参考文档来自行解包
    frame_no = req_pb.ai_req.frame_state.frame_no
    hero = req_pb.ai_req.frame_state.heroes[0]      # 只有一个英雄，所以取第一个
    organs = req_pb.ai_req.frame_state.organs       # 所有的物件的列表，包括宝箱和buff
    
    # 例子1，如果需要得到英雄的位置，可以这样获取
    pos = hero.pos
    
    # 例子2，如果需要得到英雄的加速状态和闪现技能的冷却时间，可以这样获取
    speed_up = hero.speed_up
    cooldown = hero.talent.cooldown
    
    # 特征的解包，这里提供了10个特征，具体的特征信息查看文档
    norm_pos = req_pb.ai_req.features.norm_pos
    polar_pos = req_pb.ai_req.features.polar_pos
    start_pos = req_pb.ai_req.features.start_pos
    end_pos = req_pb.ai_req.features.end_pos
    buff_pos = req_pb.ai_req.features.buff_pos
    treasure_poss = req_pb.ai_req.features.treasure_pos
    obstacle_map = list(req_pb.ai_req.features.obstacle_map)
    memory_map = list(req_pb.ai_req.features.memory_map)
    treasure_map = list(req_pb.ai_req.features.treasure_map)
    end_map = list(req_pb.ai_req.features.end_map)

    # 特征处理1：当前位置的one-hot编码
    index_x, index_z = convert_pos_to_grid_pos(pos.x, pos.z)
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[index_x], one_hot_pos_z[index_z] = 1, 1
    
    # 特征处理2： 归一化位置
    norm_pos_x = norm_pos.x
    norm_pos_z = norm_pos.z

    # 特征处理3：当前位置相对初始点位、终点点位和buff点位的信息
    start_pos_features = read_relative_position(start_pos)
    end_pos_features = read_relative_position(end_pos)
    buff_pos_features = read_relative_position(buff_pos)

    # 特征处理4：当前位置相对宝箱位置的信息
    treasure_poss_features = []
    for treasure_pos in treasure_poss:
        treasure_poss_features = treasure_poss_features + list(read_relative_position(treasure_pos))

    # 特征处理5：buff是否可收集
    buff_availability = 0
    for organ in req_pb.ai_req.frame_state.organs:
        if organ.sub_type == 2:
            buff_availability = 1
    
    # 特征处理6：闪现技能是否可使用
    talent_availability = hero.talent.status

    # 特征拼接：将所有需要的特征进行拼接作为向量特征
    observation = [norm_pos_x, norm_pos_z] + one_hot_pos_x + one_hot_pos_z + end_pos_features + [buff_availability, talent_availability] + obstacle_map + end_map + memory_map
    
    # 合法动作的标注：这里提供了两个动作，分别是行走和闪现，都合法时为[1, 1]
    legal_action = [1, 1]
    if not bool(talent_availability):
        legal_action = [1, 0]       # 当闪现动作不合法时，将闪现动作的标注为0
    
    # DEBUG时检查特征是否正确
    if CONFIG.print_predict_data:
        write_file(f'{CONFIG.print_predict_data_dir}/state_dict_{req_pb.ai_req.sgame_id}_{req_pb.ai_req.frame_no}', {
            'start_pos_features' : start_pos_features,
            'end_pos_features' : end_pos_features,
            'buff_pos_features' : buff_pos_features,
            'buff_availability' : buff_availability,
            'talent_availability' : talent_availability,
            'obstacle_map' : obstacle_map,
            'treasure_map' : treasure_map,
            'end_map' : end_map,
            'memory_map' : memory_map
        })

    return observation, legal_action, [[]]
     

def parse_from_proto_to_reward(req_pb, prev_dist_list):
    """
    该函数是奖励处理的重要函数, 主要负责：
        - 数据解包, 从req_pb获取计算奖励所需要的数据
        - 奖励计算, 根据解包的数据计算奖励
        - 奖励拼接, 将所有的奖励拼接成一个list
        - 发送game over信号, 实现训练过程中aisrv主动退出游戏
    函数的输入：
        - req_pb: battlesrv发送的proto数据
        - prev_dist_list: 上一帧的信息, 包括上一帧的位置, 相对于终点, buff, 宝箱的距离
    函数的输出：
        - reward: 奖励的list
        - curr_frame_dists: 当前帧的信息, 包括当前帧的位置, 相对于终点, buff, 宝箱的距离
        - game_over: 是否结束游戏
    """
    
    # 获取当前英雄的位置坐标
    hero = req_pb.ai_req.frame_state.heroes[0]
    pos = hero.pos
    curr_pos_x = pos.x
    curr_pos_z = pos.z
    
    # 获取当前英雄的位置相对于终点的栅格化距离
    end_pos = req_pb.ai_req.features.end_pos
    end_dist = end_pos.grid_distance
    
    # 获取当前英雄的位置相对于buff的栅格化距离
    buff_pos = req_pb.ai_req.features.buff_pos
    buff_dist = buff_pos.grid_distance

    # 获取当前英雄的位置相对于宝箱的栅格化距离
    treasure_pos = req_pb.ai_req.features.treasure_pos
    treasure_dist = [pos.grid_distance for pos in treasure_pos]
    
    # 获取英雄上一帧的位置
    prev_pos = prev_dist_list[0]
    prev_pos_x = prev_pos[0]
    prev_pos_z = prev_pos[1]
    
    # 获取英雄上一帧相对于终点，buff和宝箱的栅格化距离
    prev_end_dist = prev_dist_list[1]
    prev_buff_dist = prev_dist_list[2]
    prev_treasure_dists = prev_dist_list[3:]

    # 奖励1：相对于终点的距离奖励
    if prev_end_dist != 1:      # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
        reward_end_dist = int((prev_end_dist - end_dist) * 256)     # 逆归一化
    else:
        reward_end_dist = 0

    # 奖励2: 相对于buff的距离奖励 (同学们按需自行计算)
    reward_buff_dist = 0

    # 奖励3: 相对于宝箱的距离奖励 (同学们按需自行计算)
    reward_treasure_dist = 0

    # 奖励4：获得宝箱奖励 (同学们按需自行计算)
    reward_treasure = 0

    # 奖励5：步数惩罚 (同学们按需自行计算)
    reward_step = 0

    # 奖励6：到达终点奖励
    reward_win = 0
    if req_pb.ai_req.game_status == GameStatus.success:
        reward_win = 200

    # 奖励7：撞墙惩罚
    reward_bump = 0
    game_over = False
    if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:           # 判断是否是训练模式
        if bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z):    # 判断是否撞墙
            game_over = True                                        # 撞墙则aisrv主动结束游戏，并通过battlsrv告诉gamecore游戏结束
            reward_bump = 200                                       # 对撞墙给予一个比较大的惩罚，以便agent能够尽快学会不撞墙
        
    # 奖励8：重复探索惩罚
    memory_map = req_pb.ai_req.features.memory_map
    reward_memory = memory_map[len(memory_map)//2]                  # 如果当前位置已经探索过，则给予一个惩罚，惩罚的力度取决于探索过的次数
    
    # 奖励9：闪现奖励 (同学们按需自行计算)
    reward_flicker = 0

    # 奖励的拼接，这里提供了9个奖励，同学们按需自行拼接，也可以自行添加新的奖励
    reward = [reward_end_dist * float(REWARD_CONFIG["reward_end_dist"]),
        reward_buff_dist * float(REWARD_CONFIG["reward_buff_dist"]),
        reward_treasure_dist * float(REWARD_CONFIG["reward_treasure_dists"]),
        reward_treasure * float(REWARD_CONFIG["reward_treasure"]),
        reward_step * float(REWARD_CONFIG["reward_step"]),
        reward_bump * float(REWARD_CONFIG["reward_bump"]),
        reward_win * float(REWARD_CONFIG["reward_win"]),
        reward_memory * float(REWARD_CONFIG["reward_memory"]),
        reward_flicker * float(REWARD_CONFIG["reward_flicker"])]

    # 将当前帧的信息作为return传给下一帧
    curr_frame_dists = [[curr_pos_x, curr_pos_z]] + [end_dist, buff_dist] + treasure_dist
    
    return reward, curr_frame_dists, game_over


def load_reward_config():
    global REWARD_CONFIG
    # 在init里加载json文件
    with open("/data/projects/kaiwu-fwk/conf/gorge_walk/reward_config.json", "r") as f:
        REWARD_CONFIG = json.load(f)


def read_relative_position(rel_pos):
    """
    此函数将proto传输的相对位置特征进行拆包并处理, 返回一个长度为9的向量
        - 前8维是one-hot的方向特征
        - 最后一维是距离特征
    """
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        direction[rel_pos.direction - 1] = 1

    distance = rel_pos.grid_distance
    return direction + [distance]


def bump(a1, b1, a2, b2):
    """
    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False 

    dist = ((a1-a2)**2 + (b1-b2)**2) ** (0.5)

    return dist <= 500


def write_file(file_name, write_data):
    """
    Debug检查特征时调用, 将特征写入文件用以检查特征数据是否正确
    """
    if not file_name or not write_data:
        return

    # 注意这里按照实际格式来书写
    with open(file_name, 'w') as f:
        for key, value in write_data.items():
            f.write(f'{key}\n')
            if key in [
                'start_pos_features',
                'end_pos_features',
                'buff_pos_features',
                'buff_availability',
                'talent_availability',
            ]:
                f.write(f'{value}\n')
            elif key in [
                'obstacle_map',
                'treasure_map',
                'end_map',
                'memory_map',
            ]:
                sub_lists = [value[i:i+20] for i in range(0, len(value), 20)]
                for s in sub_lists:
                    f.write(f'{s}\n')