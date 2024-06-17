import pickle
import random

import numpy as np
import pandas as pd

import gym
from gym import spaces

# 1.use_idol_vehicle=True, use_rsu=True
# 2.use_idol_vehicle=True, use_rsu=False
# 3.use_idol_vehicle=True, use_rsu=True

# np.random.seed(1)

class TaskOffloadingEnv(gym.Env):

    def __init__(self,
                 n_ue=5,
                 n_bs=2,
                 n_en=10,
                 n_task=5,
                 use_en=True,
                 use_bs=True,
                 map_size=200):
        """
        :param n_ue    用户的数量 K
        :param n_bs    基站数量      N
        :param n_en    闲置用户的数量 H
        :n_task        总任务数量    M
        :use_en        是否卸载到闲置用户，默认为真
        :use_bs        是否卸载到基站，默认为真
        :map_size      地图大小，默认为100x100
        """
        super(TaskOffloadingEnv, self).__init__()

        self.use_en = use_en
        self.use_bs = use_bs

        # 定义闲置用户、用户、基站和任务的数量
        self.n_en = self.H = n_en if self.use_en else 0
        self.n_ue = self.K = n_ue
        self.n_bs = self.N = n_bs if self.use_bs else 0
        self.n_task_per_ue = self.M = n_task

        # 定义地图的边长
        self.map_size = map_size

        # 定义每一个任务 task_{ij} 的属性，每一个任务包含三个属性，分别是：
        # 
        # [D_{ij},  t_{ij}, Y_{ij}]
        # 1. D_{ij}：计算所需资源量
        # 2. t_{ij}：任务完成最大时延
        # 3. Y_{ij}：任务数据大小
        # 4. 产生任务的类型  （没有实现）
        # TODO 需要修改 改成三个属性
        # TODO 需要确定数值范围合理性
        self.tasks_prop = np.random.randint(0, 10, size=(self.n_task_per_ue, 3))   #[[7 1 2 4]
                                                                                        # [1 8 3 3]
                                                                                        # [5 9 0 6]
                                                                                        # [2 3 6 9]
                                                                                        # [1 4 8 5]]

        # 当前执行任务的用户
        self.cur_ue = 0
        # 定义当前执行任务的索引号
        self.cur_task = 0

        # 维护当前可卸载的设备信息
        # self.mask_vehicle = np.ones(shape=self.n_idol_vehicle, dtype=np.uint8)
        # self.mask_rsu = np.ones(shape=self.n_rsu, dtype=np.uint8)

        # 维护闲置车辆和路边单元设备的计算资源暂用情况  初始化为1（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)
        self.resource_bs = np.ones(self.n_bs)

        # 当前设备最大计算能力
        # TODO 需要确定数值范围合理性
        self.resource_ue_max = np.ones(shape=self.n_ue) * 0.5  # GHZ
        self.resource_en_max = np.ones(self.n_en) * 2  # GHZ
        self.resource_bs_max = np.ones(shape=self.n_bs) * 4     # GHZ


        # 车辆卸载任务时的上传传输数据功率（仅仅对车辆） 30mW
        self.idol_vehicle = np.ones(self.n_ue) * 10
        self.upload_data_ue = np.ones(self.n_ue) * 10 * 3

        # 维护当前设备的位置（假设是网格环境）
        # self.position_vehicle = np.random.randint(0, 10, size=(self.n_vehicle, 2))
        # self.position_idol_vehicle = np.random.randint(0, 10, size=(self.n_idol_vehicle, 2))
        # self.position_rsu = np.random.randint(0, 10, size=(self.n_rsu, 2))

        self.bandwidth_ue = np.ones(self.n_ue)

        # 最大带宽
        self.MAX_BANDWIDTH_BE = 5000
        self.MAX_BANDWIDTH_BR = 10000
        # 噪声功率谱密度
        self.N0 = 10

        # 任务分得带宽百分率
        self.task_percent_bandwidth = np.random.random(size=1)
        # 任务传输功率 mW
        self.task_p = np.random.randint(10, 100, size=self.n_ue)
        # 任务传输信道增益  dB
        # TODO 这里由于传输设备和接收设备之间的信道增益不太一样可能需要修改
        self.task_h_v = np.random.randint(0, 10, size=self.n_en)
        self.task_h_r = np.random.randint(10, 100, size=self.n_bs)

        # 任务最大等待时长
        self.task_time = np.random.randint(4, 8, size=self.n_task_per_ue)

        self.vehicle_v = 1

        self.R0 = 200
        self.R1 = 3000

        self.lambda1 = 1
        #self.lambda2 = 1

        self.beta1 = 1
        self.beta2 = 1
        self.beta3 = 1

        self.o_ij = 1
        self.w_ij = 1
        self.u_ij = 1

        #self.z = 1e-11
        self.z = 2

        # 状态空间，四行分别是：
        # 1. 任务三个属性
        # 2. 标记卸载到自身
        # 3. 可卸载设备集合标记
        # 4. 卸载设备计算资源占用
        # 5. 卸载设备计算能力
        # 6. 卸载设备位置信息
        # self.n_idol_vehicle + self.n_rsu + 1 + \
        self.observation_space = \
            3 + \
            self.n_en + self.n_bs + 1 + \
            self.n_en + self.n_bs + 1 + \
            self.n_en * 2 + self.n_bs * 2 + 2
        # NOTE: 接入 openai gym 接口
        self.observation_space = spaces.Discrete(self.observation_space)

        # 动作空间，分别是
        # 1：卸载到自身
        # self.n_idol_vehicle：卸载到闲置车辆
        # self.n_rsu：卸载到路边单元
        # 2：一个带宽占比，一个计算资源占比
        self.action_space = 1 + self.n_en + self.n_bs
        # NOTE: 接入 openai gym 接口
        self.action_space = spaces.Discrete(self.action_space)




    def reset(self):
        # np.random.seed(1)
        """
        随机初始化
        """
        # 定义每一个任务 task_{ij} 的属性，每一个任务包含三个属性，分别是：
        # 
        # [D_{ij},  t_{ij}, Y_{ij}]
        # 1. D_{ij}：计算所需资源量
        # 2. t_{ij}：任务完成最大时延
        # 3. Y_{ij}：任务数据大小
        # 4. 产生任务的类型
        # [900, 1100]
        self.tasks_comp = np.random.randint(20, 25, size=(self.n_task_per_ue, 1))
        # self.tasks_comp = np.array([[1053],[1093],[1010],[1074],[ 913],[1096],[ 923],[1067],[1035],[1084]])
        #rint(self.tasks_comp)
        # 10ms
        self.tasks_dely = np.full(shape=(self.n_task_per_ue, 1), fill_value=10)
        #self.tasks_dely = np.array([[10],[10],[10],[10],[ 913],[1096],[ 923],[1067],[1035],[1084]])
        # [300, 500] kbit
        self.tasks_size = np.random.randint(300, 500, size=(self.n_task_per_ue, 1))
        # self.tasks_size = np.array( [[331],[332],[353],[301],[321],[428],[498],[405],[309],[406]])
        # 从已有的任务中生成任务类型
        #self.tasks_type = np.random.randint(0,  2, size=(self.n_task_per_vehicle, 1))
        # self.tasks_type = np.array([[0],[1],[0],[1],[1],[0],[1],[0],[1],[0]])
            # self._generate_task_type(
            # self.n_task_per_vehicle)  #
        #print(self.tasks_type)
        self.tasks_prop = np.concatenate((self.tasks_comp, self.tasks_dely, self.tasks_size), axis=1)

        # 维护闲置车辆和路边单元设备的计算资源暂用情况（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)
        self.resource_bs = np.ones(self.n_bs)

        # 当前设备最大计算能力
        self.resource_ue_max = np.ones(shape=self.n_ue) * 0.5  # GHZ
        self.resource_en_max = np.ones(self.n_en) * 2  # GHZ
        self.resource_bs_max = np.ones(shape=self.n_bs) * 4  # GHZ


        # 车辆卸载任务时的上传传输数据功率（仅仅对车辆） 30mW
        #self.idol_vehicle = np.ones(self.n_vehicle) * 10
        self.upload_data_ue = np.ones(self.n_ue) * 10 * 3

        # 维护当前设备的位置（假设是网格环境），网格环境[3000 * 3000]m
        self.position_ue = np.random.randint(0, self.map_size, size=(self.n_ue, 2))
        self.position_en = np.random.randint(0, self.map_size, size=(self.n_en, 2))
        self.position_bs = np.random.randint(0, self.map_size, size=(self.n_bs, 2))

        # 最大带宽
        self.MAX_BANDWIDTH_BV = 5000   # khz
        self.MAX_BANDWIDTH_BR = 10000  # khz
        # 噪声功率谱密度
        self.N0 = 10

        # 任务传输信息增益
        self.task_h = np.random.randint(1, 10, size=1)

        # 任务分得带宽百分率
        self.task_percent_bandwidth = np.random.random(size=1)
        # 任务传输功率
        self.task_p = np.random.randint(10, 100, size=self.n_ue)

        # 任务传输信道增益
        self.task_h_v = np.random.randint(0, 10, size=1)
        self.task_h_r = np.random.randint(10, 100, size=1)


        # 记录闲置设备和路边单元能否卸载[0,1]
        self.mask_en = np.zeros(shape=self.n_en)
        self.mask_bs = np.zeros(shape=self.n_bs)


        self.cur_ues = np.random.randint(0, self.n_ue, size=self.n_task_per_ue)
        self.cur_task = 0
        self.cur_ue = self.cur_ues[self.cur_task]

        state = self._state()

        self.cur_task = 0

        return state
    # 0 不能卸载到自身 1 可以
    def get_mask_action(self):
        return np.concatenate([np.array([1]), self.mask_en, self.mask_bs], axis=-1)

    def step(self, action_mix):
        """
        action: {'id': discrete_action, 'parameter': continuous_action}
        返回下一个状态，以及奖励 next_state, reward, done
        
        """
        t_comm_percent = t_comp_percent = 0.8
        action = action_mix
        r_ = self._reward(action, t_comm_percent, t_comp_percent)
        r_ = np.clip(r_, -1000, np.inf)

        self._update_state(action, t_comm_percent, t_comp_percent)

        state_ = self._state()

        return state_, np.array(r_), self.cur_task + 1 == len(self.tasks_prop), {} #可返回

    def _state(self):
        """根据当前环境变量，计算环境状态
        Returns:
            _type_: _description_
        """

        # 计算可卸载的闲置车辆
        for i in range(self.n_en):
            # 距离
            d0 = np.sqrt(np.sum(np.power(
                self.position_en[i] - self.position_ue[self.cur_ue], 2)))

            task_pro = self.tasks_prop[self.cur_task]

            R_ij = self.MAX_BANDWIDTH_BV / self.K * \
                   np.log(1 + self.task_p[self.cur_ue] * self.task_h_v /
                          (self.N0 * self.MAX_BANDWIDTH_BV / self.K))


            t_ij = task_pro[0] / (self.resource_en[i] * self.resource_en_max[i] + 1e-5) + \
                   task_pro[2] / (R_ij + 100)
            # TODO 距离有问题
            x = self.position_ue[self.cur_ue][0] + self.vehicle_v * t_ij
            x_vehicle = self.position_en[i][0]

            y = self.position_ue[self.cur_ue][1] + self.vehicle_v * t_ij
            y_vehicle = self.position_en[i][1]

            l0 = np.sqrt(np.power(x - x_vehicle, 2) + np.power(y - y_vehicle, 2))

            if self.mask_en[i] == np.inf:
                self.mask_en[i] = 0
                continue
            if d0 < self.R0 and l0 < self.R0:
                self.mask_en[i] = 1

        # 计算可卸载的路边单元
        for i in range(self.n_bs):
            d1 = np.sqrt(np.sum(np.power(self.position_bs[i] - self.position_ue[self.cur_ue], 2)))

            task_pro = self.tasks_prop[self.cur_task]

            R_ij = self.MAX_BANDWIDTH_BR  / self.K * \
                   np.log(1 + self.task_p[self.cur_ue] * self.task_h_r /
                          (self.N0 * self.MAX_BANDWIDTH_BR / self.K))
            t_ij = task_pro[0] / (self.resource_bs[i] * self.resource_bs_max[i] + 1e-5)+\
                   task_pro[2] /(R_ij)

            x = self.position_ue[self.cur_ue][
                    0] + self.vehicle_v * t_ij
            x_vehicle = self.position_bs[i][0]

            y = self.position_ue[self.cur_ue][1] + self.vehicle_v * t_ij
            y_vehicle = self.position_bs[i][1]

            l1 = np.sqrt(np.power(x - x_vehicle, 2) + np.power(y - y_vehicle, 2))

            if self.mask_bs[i] == np.inf:
                self.mask_bs[i] = 0
                continue
            if d1 < self.R1 and l1 < self.R1:
                self.mask_bs[i] = 1

        # 连接所有属性
        # np.array([self.resource_vehicle_max[self.cur_vehicle]]),
        # self.resource_idol_vehicle_max,
        # self.resource_rsu_max,

        return np.concatenate((self.tasks_prop[self.cur_task],
                               np.array([1]),
                               self.mask_en,
                               self.mask_bs,
                               np.array([self.resource_ue[self.cur_ue]]),
                               self.resource_en,
                               self.resource_bs,
                               np.array(self.position_ue[self.cur_ue]),
                               self.position_en.reshape(self.position_en.size),
                               self.position_bs.reshape(self.position_bs.size)))

    def _reward_idol_vehicle(self, action, t_comm_percent, t_comp_percent):

        """将任务卸载到闲置车辆获得的奖励

        Args:
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_

        Returns:
            _type_: _description_
        """
        task_prop = self.tasks_prop[self.cur_task]

        # 任务卸载速率
        R_ij = t_comm_percent * self.MAX_BANDWIDTH_BV / self.K * \
               np.log2(1 + self.task_p[self.cur_ue] * self.task_h_v /
                       (self.N0 * self.MAX_BANDWIDTH_BV / self.K))



        t_ij = task_prop[2] / (R_ij) + \
               task_prop[0] / (t_comp_percent * self.resource_en_max[action]+ 1e-5)
        t_ij = self.w_ij * t_ij

        # 时延效用（希望剩余时间预算越多约好）
        G_t = self.lambda1 * (task_prop[1] - t_ij)

        # 卸载能耗
        E_v = self.upload_data_ue[self.cur_ue] * task_prop[2] /(R_ij + 1e-5) + \
              self.idol_vehicle[self.cur_ue] * task_prop[0] / \
              (t_comp_percent * self.resource_en_max[action] + 1e-5)

        # 能耗效用（希望能耗越少约好）
        G_e = self.o_ij * self.beta1 * E_v
        # print("en:",G_t,G_e/100)
        return float(G_t - G_e)

    def _reward_rsu(self, action, t_comm_percent, t_comp_percent):
        """将任务卸载到路边单元获得的奖励

        Args:
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_

        Returns:
            _type_: _description_
        """
        t_comm_percent = t_comp_percent = 0.8
        task_prop = self.tasks_prop[self.cur_task]
        # 任务卸载速率
        R_ij = t_comm_percent * self.MAX_BANDWIDTH_BR / self.K * \
               np.log2(1 + self.task_p[self.cur_ue] * self.task_h_r /
                       (self.N0 * self.MAX_BANDWIDTH_BR / self.K))



        t_ij = task_prop[2] / (R_ij) + \
               task_prop[0] / (t_comp_percent * self.resource_bs_max[action] + 1e-5)
        t_ij = self.u_ij * t_ij

        # 时延效用（希望剩余时间预算越多约好）
        G_t = self.lambda1 * (task_prop[1] - t_ij)
        # G_t = self.lambda1 * ( - t_ij)
        #print(G_t)
        # 卸载能耗
        # TODO 修改RSU的计算能力
        E_v = self.upload_data_ue[self.cur_ue] * task_prop[2] / (R_ij + 1e-5) + \
              self.idol_vehicle[self.cur_ue] * task_prop[0] / \
              (t_comp_percent * self.resource_bs_max[action] + 1e-5)
        #print(E_v)
        # 能耗效用（希望能耗越少约好）
        G_e = self.u_ij * self.beta3 * E_v
        #print('bs:',float(G_t - G_e))
        # print("rsu:",G_t,G_e)
        return float(G_t - G_e)

    def _reward_self(self, action, t_comm_percent, t_comp_percent):
        """将任务卸载到zishen获得的奖励

        Args:
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_

        Returns:
            _type_: _description_
        """
        task_prop = self.tasks_prop[self.cur_task]

        t_ij = task_prop[0] / (t_comp_percent * self.resource_ue_max[self.cur_ue]+ 1e-5)
        t_ij = self.o_ij * t_ij
        G_t = self.lambda1 * (task_prop[1] - t_ij)
        # 卸载能耗
        # TODO 修改了卸载到本地计算能耗
        # E_v = self.z * task_prop[0] / self.resource_ue_max[self.cur_ue]
        E_v = self.z * task_prop[0] * self.resource_ue_max[self.cur_ue] ** 2

        # 能耗效用（希望能耗越少约好）
        G_e = self.o_ij * self.beta1 * E_v
        # print("SELF:",G_t,G_e)
        return float(G_t - G_e)

    def _reward(self, action, t_comm_percent, t_comp_percent):
        if action == 0:
            return self._reward_self(action, t_comm_percent, t_comp_percent)
        else:
            if self.use_en and self.use_bs:
                if 0 < action <= self.n_en:
                    action -= 1
                    return self._reward_idol_vehicle(action, t_comm_percent, t_comp_percent)
                else:
                    action -= self.n_en + 1
                    return self._reward_rsu(action, t_comm_percent, t_comp_percent)
            if self.use_en:
                action -= 1
                return self._reward_idol_vehicle(action, t_comm_percent, t_comp_percent)
            if self.use_bs:
                action -= 1
                return self._reward_rsu(action, t_comm_percent, t_comp_percent) - 150


    # def _reward_outdate(self, action, t_comm_percent, t_comp_percent):
    #     G_t = 0
    #     for i , j in itertools.product(range(self.K), range(self.M)):
    #         task_prop = self.tasks_prop[j]
    #
    #         R_ij = self.MAX_BANDWIDTH_BV / self.K * \
    #             np.log(1 + self.task_p * self.task_h /
    #                    (self.N0 * self.MAX_BANDWIDTH_BV / self.K))
    #         t_ij = task_prop[0] / (self.resource_idol_vehicle[i] * self.resource_vehicle_max[i] + 1e-5) + \
    #             task_prop[2] / (R_ij + 1e-5)
    #
    #         R_ij = self.MAX_BANDWIDTH_BR / self.K * \
    #             np.log(1 + self.task_p * self.task_h /
    #                    (self.N0 * self.MAX_BANDWIDTH_BR / self.K))
    #         t_ij_safe = task_prop[0] / (self.resource_idol_vehicle[i] * self.resource_vehicle_max[i] + 1e-5) + \
    #             task_prop[2] / (R_ij + 1e-5)
    #
    #
    #         G_t += self.lambda1 * (self.task_time[j] - t_ij) + \
    #             self.lambda2 * (self.task_time[j] - t_ij_safe)
    #
    #
    #     G_e = 0
    #     for i , j in itertools.product(range(self.K), range(self.M)):
    #         task_prop = self.tasks_prop[j]
    #
    #         # E_ij^local = z * D_ij (C_local)^2
    #         E_local = self.z * task_prop[0] * self.resource_idol_vehicle[i] ** 2
    #
    #         R_ij = self.MAX_BANDWIDTH_BV / self.K * \
    #             np.log(1 + self.task_p * self.task_h /
    #                    (self.N0 * self.MAX_BANDWIDTH_BV / self.K))
    #
    #         E_v = self.upload_data_vehicle[i] * task_prop[2] / (R_ij + 1e-5) + \
    #             self.idol_vehicle[i] * task_prop[0] / (self.resource_idol_vehicle[i] * self.resource_vehicle_max[i] + 1e-5)
    #
    #         R_ij = self.MAX_BANDWIDTH_BR / self.K * \
    #             np.log(1 + self.task_p * self.task_h /
    #                    (self.N0 * self.MAX_BANDWIDTH_BR / self.K))
    #         E_r = self.upload_data_vehicle[i] * task_prop[2] / (R_ij + 1e-5) + \
    #             self.idol_vehicle[i] * task_prop[0] / (self.resource_rsu[i] * self.resource_rsu_max[i])
    #
    #         G_e += self.o_ij * self.beta1 * E_local + \
    #                self.w_ij * self.beta2 * E_v + \
    #                self.u_ij * self.beta3 * E_r
    #
    #     return G_t - G_e

    def _update_state(self, action, t_comm_percent, t_comp_percent):
        """根据外部采取的动作更新当前环境状态

        Args:
            action (_type_): _description_
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_
        """
        self.mask_en = np.zeros(shape=self.n_en)
        self.mask_bs = np.zeros(shape=self.n_bs)

        # 重置通信资源
        self.bandwidth_vehicle = np.ones(self.n_ue)
        # 分配通信资源
        self.bandwidth_vehicle[self.cur_ue] -= t_comm_percent

        # 判断动作是卸载到闲置策略还是路边单元
        # self._update_state(action, t_comm_percent, t_comp_percent)
        if action == 0:
            self.resource_ue = np.ones(self.n_ue)
            self.resource_ue[self.cur_ue] -= t_comp_percent
        if self.use_en and self.use_bs:
            if action <= self.n_en:
                action -= 1
                self.resource_en = np.ones(self.n_en)
                self.resource_en[action] -= t_comp_percent
                self.mask_en[action] = np.inf
            else:
                action -= self.n_en + 1
                self.resource_bs = np.ones(self.n_bs)
                self.resource_bs[action] -= t_comp_percent
                self.mask_bs[action] = np.inf
        elif self.use_en:
            action -= 1
            self.resource_en = np.ones(self.n_en)
            self.resource_en[action] -= t_comp_percent
            self.mask_en[action] = np.inf
        else:
            action -= 1
            self.resource_bs = np.ones(self.n_bs)
            self.resource_bs[action] -= t_comp_percent
            self.mask_bs[action] = np.inf

        # 更新系统当前任务车辆和任务索引号
        self.cur_task += 1
        self.cur_ue = self.cur_ues[self.cur_task] # int(np.random.randint(0, self.n_vehicle, size=1))


class TaskOffloadingEnvActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(TaskOffloadingEnvActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(np.float32(old_as.spaces[1].spaces[i].low), np.float32(old_as.spaces[1].spaces[i].high), dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action


if __name__ == "__main__":
    env = TaskOffloadingEnv()

    state = env.reset()

    for _ in range(100):
        a = np.random.randint(0, 10, size=1)
        bandwidth = np.random.random(1)
        compute = np.random.random(1)

        state_, reward, done, info = env.step(np.array([[0, 1, 2], [[0.4, 0.3], [0.4, 0.3], [0.4, 0.3]]], dtype=int))
        if done: break
        print(state_)
