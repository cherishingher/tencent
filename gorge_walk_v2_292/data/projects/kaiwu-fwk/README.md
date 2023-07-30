**KaiwuDRL** 是腾讯王者荣耀开悟团队自研的深度强化学习框架, 集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体, 旨在帮助越来越多的使用者完成AI赋能，实现产业智能化升级。

## 欢迎来到开悟

### 赛题介绍：

**场景：** 峡谷漫步v2

**地图：** 峡谷之森

**阵容：** 鲁班七号

**支持框架：** PyTorch

**支持算法：** DQN，PPO，用户自定义算法（DIY）

### 代码目录介绍

**app/gorge_walk:** 

- **algorithm：** 强化学习中的模型和算法，其中torch_network.py实现了神经网络模型，torch_xxx_model.py实现了不同的强化学习算法。
- **sample_processor：** 样本处理逻辑，不同算法所需要的样本是不同的，重点关注`gorge_walk_sample_processor.py`里的`gen_expr()`和`proc_exprs()`两个函数
- **state_aciton_reward：** 强化学习中的状态，动作，以及奖励，支持用户自定义state和reward，重点关注`gorge_walk_obs_reward.py`里的`parse_from_proto_to_state()`以及`parse_from_proto_to_rewarad()`两个函数
- **train_test.py：** 进行代码调试的文件

**conf：** 配置文件，主要包含强化学习训练相关的一些配置参数

- **gorge_walk：** 算法和场景相关的配置，重点关注`config.py`


💡[点此查看峡谷漫步v2开发指南](https://doc.aiarena.tencent.com/kaiwu-arena/gorgewalk_v2/comp/latest/)

