#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_base_model.py
@Author  :kaiwu
@Date    :2022/12/15 22:50 

'''


import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
import re
import os
from framework.common.utils.common_func import get_first_line_and_last_line_from_file
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from conf.gorge_walk.config import Config


class BaseModel(object):
    """
        base_model, 模型的实现: 包括神经网络、模型预测、模型训练、模型保存、模型恢复
        dqn, ppo, diy等继承base_model
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
        super().__init__()
        self.file_queue = []

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

    def should_stop(self):
        return False

    def stop(self):
        return True

    def load_last_new_model(self, models_path):
        """
            Description: 根据传入的模型路径，载入最新模型
        """
        checkpoint_file = f'{models_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        _, last_line = get_first_line_and_last_line_from_file(checkpoint_file)
        if not last_line:
            return

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = last_line.split(f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1]
        checkpoint_id = re.findall(r'\d+\.?\d*', checkpoint_id)[0]

        self.load_param(path=models_path, id=checkpoint_id)

    def load_specific_model(self, models_path):
        """
            Description: 根据传入的模型，载入指定模型
        """
        checkpoint_id = models_path.split(f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1][:-4]
        models_path = os.path.dirname(models_path)

        self.load_param(path=models_path, id=checkpoint_id)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def save_param(self, path=None, id='1'):
        """
            Description: 保存模型的方法
            ----------

            Parameters
            ----------
            path: str
                保存模型的路径
            id: int
                保存模型的id
        """
        path = f'{CONFIG.restore_dir}/{self.name}/'

        torch.save(self.model.state_dict(), f"{str(path)}/model.ckpt-{str(id)}.pkl")
        file_exist_flag = os.path.exists(f"{str(path)}/checkpoint")
        with open(f"{str(path)}/checkpoint", mode='a') as fp:
            if not file_exist_flag:
                fp.writelines([
                    f"checkpoints list\n"
                ])
            fp.writelines([
                f"all_model_checkpoint_paths: \"{str(path)}/model.ckpt-{str(id)}\"\n"
            ])
        self.add_file_to_queue(f"{str(path)}/model.ckpt-{str(id)}.pkl")

    def add_file_to_queue(self, file_path):
        self.file_queue.append(file_path)
        if len(self.file_queue) > Config.MAX_FILE_KEEP_CNT:
            to_delete_file = self.file_queue.pop(0)
            if os.path.exists(to_delete_file):
                os.remove(to_delete_file)

    def load_param(self, path='/tmp/pyt-model', id='1'):
        self.model.load_state_dict(torch.load(f"{str(path)}/model.ckpt-{str(id)}.pkl",
                                   map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
