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
                 n_en=5,
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

        # 定义闲置工人和无人机和任务的数量
        self.n_ue = self.K = n_ue
        self.n_en = self.H = n_en
        self.n_task_per_ue = self.M = n_task

        # 定义地图的边长
        self.map_size = map_size

        # 任务坐标x y
        self.tasks_p = np.random.randint(0, 200, size=(self.n_task_per_ue, 2))
        # 任务存活时间
        self.tasks_time = np.random.randint(0, 10, size=(self.n_task_per_ue, 1))
        # 任务优先级
        self.task_priority = np.random.randint(0, 10, size=(self.n_task_per_ue, 1))
        # 任务基础报酬
        self.tasks_pay = np.random.randint(30, 80, size=(self.n_task_per_ue, 1))
        # 组合起来
        self.tasks_prop = np.concatenate((self.tasks_p, self.tasks_time, self.task_priority, self.tasks_pay),
                                         axis=1)

        # 当前执行任务的用户
        self.cur_ue = 0
        # 定义当前执行任务的无人机
        self.cur_en = 0
        # 定义当前执行任务的索引号
        self.cur_task = 0

        # 维护工人和无人机的计算资源暂用情况  初始化为1（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        # 工人属性[x,y,报酬]
        self.position_ue = np.random.randint(0, 200, size=(self.n_ue, 2))
        self.pay_ue = np.random.randint(5, 10, size=(self.n_ue, 1))
        self.ue_prop = np.hstack((self.position_ue, self.pay_ue))

        # 无人机属性[x,y,无人机能耗]
        self.position_en = np.random.randint(0, 200, size=(self.n_en, 2))
        self.pay_en = np.random.randint(0, 5, size=(self.n_en, 1))
        self.en_prop = np.hstack((self.position_en, self.pay_en))

        # 状态空间，四行分别是：
        # 任务属性 [x, y, 截至时间, 优先级, 基础报酬]
        # 当前执行任务的用户和无人机索引号
        # 工人和无人机的属性 [x, y, 报酬/无人机能耗]
        self.observation_space = \
        25 + \
        self.n_ue + self.n_en + \
        self.n_ue * 3 + self.n_en * 3 + 10

        # NOTE: 接入 openai gym 接口
        self.observation_space = spaces.Discrete(self.observation_space)
        self.action_space =  self.n_en + self.n_ue + 2 *(self.n_en + self.n_ue)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.n_ue + self.n_en),  # 离散动作部分
            # spaces.Box(low=0, high=100, shape=(self.n_en, 1), dtype=np.float32),  # 无人机移动速度的连续部分
            # spaces.Box(low=0, high=20, shape=(self.n_ue, 1), dtype=np.float32) , # 工人移动速度的连续部分
            spaces.Tuple(
                tuple( spaces.Box(low=np.array([0, 0]), high=np.array([100, 100]), dtype=np.float32)for _ in range(self.n_en))
            ),

            spaces.Tuple(
                tuple(spaces.Box(low=np.array([0, 0]), high=np.array([100, 20]), dtype=np.float32) for _ in
                      range(self.n_ue))
            ),
        )
        )

        # 动作空间，分别是
        # self.n_en 分配给工人
        # self.n_bs：分配给无人机
        # 2：一个工人速度 一个无人机速度
        #self.action_space = self.n_en + self.n_ue + 2
        # NOTE: 接入 openai gym 接口
        #self.action_space = spaces.Tuple((
            #spaces.Discrete(self.action_space - 2),
            #spaces.Tuple(
                #tuple(spaces.Box(low=np.array([-1, -1]), high=np.array([-1, 1]), dtype=np.float32)
                      #for _ in range(self.n_en))  # 无人机的移动速度的连续部分
            #),
            #spaces.Tuple(
                #tuple(spaces.Box(low=np.array([-1, -1]), high=np.array([-1, 1]), dtype=np.float32)
                      #for _ in range(self.n_ue))  # 工人的移动速度的连续部分
            #)
        #))

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
        # 任务属性 [x, y, 截至时间, 优先级, 基础报酬]
        # 任务坐标x,y
        self.tasks_p = np.random.randint(0, 200, size=(self.n_task_per_ue, 2))
        # 任务截至时间
        self.tasks_time = np.random.randint(0, 10, size=(self.n_task_per_ue, 1))
        # 任务优先级
        self.task_priority = np.random.randint(0, 10, size=(self.n_task_per_ue, 1))
        # 任务基础报酬
        self.tasks_pay = np.random.randint(0, 10, size=(self.n_task_per_ue, 1))
        # 组合起来
        self.tasks_prop = np.concatenate((self.tasks_p, self.tasks_time, self.task_priority,self.tasks_pay), axis=1)

        # 维护闲置车辆和路边单元设备的计算资源暂用情况（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        # 工人属性[x,y,报酬]
        self.position_ue = np.random.randint(0, 200, size=(self.n_ue, 2))
        self.pay_ue = np.random.randint(5, 10, size=(self.n_ue, 1))
        self.ue_prop = np.hstack((self.position_ue, self.pay_ue))

        # 无人机属性[x,y,无人机能耗]
        self.position_en = np.random.randint(0, 200, size=(self.n_en, 2))
        self.pay_en = np.random.randint(0, 5, size=(self.n_en, 1))
        self.en_prop = np.hstack((self.position_en, self.pay_en))

        # 记录工人和无人机能否卸载[0,1] 设置1 or 0 ？
        self.mask_ue = np.zeros(shape=self.n_ue)
        self.mask_en = np.zeros(shape=self.n_en)

        # 当前执行任务的用户
        self.cur_ue = 0
        # 定义当前执行任务的无人机
        self.cur_en = 0
        # 定义当前执行任务的索引号
        self.cur_task = 0

        state = self._state()

        self.cur_task = 0

        return state
    # 0 不能卸载到自身 1 可以
    def get_mask_action(self):
        return np.concatenate([np.array([1]), self.mask_ue, self.mask_en], axis=-1)

    def step(self, action_mix):
        """
        action: {'id': discrete_action, 'parameter': continuous_action}
        返回下一个状态，以及奖励 next_state, reward, done
        
        """
        action = action_mix[0]
        print("action:", action)
        parameters = action_mix[1][action]

        # action = action_mix[0]
        # NOTE：这里可以根据需要改为 action.t_comm_percent 和 action.t_comp_percent
        t_comm_percent = parameters[0]
        t_comp_percent = parameters[1]

        # action = 0 if action == 0 else \
        #     action - 1 if action >= 1 and action <= self.n_idol_vehicle else \
        #     action - 1 - self.n_idol_vehicle

        t_comm_percent = max(min(t_comm_percent, 1), 0)
        t_comp_percent = max(min(t_comp_percent, 1), 0)

        # r_ = self._reward(action, t_comm_percent, t_comp_percent)
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

        # 计算分配的工人
        for i in range(self.n_ue):
            # 计算工人与当前任务的距离
            d0 = np.sqrt(np.sum(np.power(
                self.position_ue[i] - self.tasks_p[self.cur_task], 2)))
            # 获取当前任务属性
            task_pro = self.tasks_prop[self.cur_task]
            # 计算任务完成时间 这里假设工人速度为5
            t_ij = d0/5
            # 如果时间超过截至时间 就不使用
            if t_ij < task_pro[2]:
                self.mask_ue[i] = 1

        # 计算可卸载的无人机
        for i in range(self.n_en):
            # 计算工人与当前任务的距离
            d0 = np.sqrt(np.sum(np.power(
                self.position_en[i] - self.tasks_p[self.cur_task], 2)))
            # 获取当前任务属性
            task_pro = self.tasks_prop[self.cur_task]
            # 计算任务完成时间 这里假设无人机速度为20
            t_ij = d0 / 20
            # 如果时间超过截至时间 就不使用
            if t_ij < task_pro[2]:
                self.mask_en[i] = 1

        # 连接所有属性
        return np.concatenate((
            self.tasks_prop[self.cur_task],  # 当前任务属性
            np.array([1]),  # 标识符
            self.mask_ue,  # 工人掩码
            self.mask_en,  # 无人机掩码
            np.array([self.resource_ue[self.cur_ue]]),  # 当前用户资源使用情况
            np.array([self.resource_en[self.cur_en]]),  # 无人机资源使用情况
            self.position_ue.reshape(-1),  # 所有工人位置
            self.position_en.reshape(-1),  # 所有无人机位置
            self.pay_ue.reshape(-1),  # 所有工人报酬
            self.pay_en.reshape(-1)  # 所有无人机能耗
        ))

    def _reward_ue(self, action, t_comm_percent, t_comp_percent):
        """将任务分配给工人获得的奖励

        Args:
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 获取当前任务属性
        task_prop = self.tasks_prop[self.cur_task]
        # 获取工人属性
        ue_prop = self.ue_prop[self.cur_ue]
        # 计算工人与任务之间距离
        d0 = np.sqrt(np.sum(np.power(
            self.position_ue[self.cur_ue] - self.tasks_p[self.cur_task], 2)))
        t_ij = d0 / 5
        get = task_prop[3]*task_prop[4]
        pay = (task_prop[2]-t_ij)+ue_prop[2]
        return float(get-pay)

    def _reward_en(self, action, t_comm_percent, t_comp_percent):
        """将任务分配给无人机获得的奖励

        Args:
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 获取当前任务属性
        task_prop = self.tasks_prop[self.cur_task]
        # 获取工人属性
        en_prop = self.ue_prop[self.cur_en]
        # 计算无人机与任务之间距离
        d0 = np.sqrt(np.sum(np.power(
            self.position_en[self.cur_en] - self.tasks_p[self.cur_task], 2)))
        t_ij = d0 / 20
        get = task_prop[3]*task_prop[4]+(task_prop[2]-t_ij)
        pay = en_prop[2]+t_ij*5
        return float(get-pay)

    def _reward(self, action, t_comm_percent, t_comp_percent):
        if 0 < action <= self.n_ue:
            action -= 1
            return self._reward_ue(action, t_comm_percent, t_comp_percent)
        else:
            action -= self.n_ue + 1
            return self._reward_en(action, t_comm_percent, t_comp_percent)



    def _update_state(self, action, t_comm_percent, t_comp_percent):
        """根据外部采取的动作更新当前环境状态

        Args:
            action (_type_): _description_
            t_comm_percent (_type_): _description_
            t_comp_percent (_type_): _description_
        """
        self.mask_ue = np.zeros(shape=self.n_ue)
        self.mask_en = np.zeros(shape=self.n_en)


        # 判断动作是卸载到工人还是无人机
        if action <= self.n_ue:
            action -= 1
            self.resource_ue = np.ones(self.n_ue)
            self.resource_ue[action] -= t_comp_percent
            self.mask_ue[action] = np.inf
        else:
            action -= self.n_en + 1
            self.resource_en = np.ones(self.n_en)
            self.resource_en[action] -= t_comp_percent
            self.mask_en[action] = np.inf

        # 更新系统当前任务车辆和任务索引号
        self.cur_task += 1
        #self.cur_ue = self.cur_ues[self.cur_task] # int(np.random.randint(0, self.n_vehicle, size=1))


class TaskOffloadingEnvActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    #def __init__(self, env):
        #super(TaskOffloadingEnvActionWrapper, self).__init__(env)
        #old_as = env.action_space
        #num_actions = old_as.spaces[0].n
        #self.action_space = gym.spaces.Tuple((
            #old_as.spaces[0],  # actions
            #*(gym.spaces.Box(np.float32(old_as.spaces[1].spaces[i].low), np.float32(old_as.spaces[1].spaces[i].high), dtype=np.float32)
              #for i in range(0, num_actions))
        #))

    # def __init__(self, env):
    #     super(TaskOffloadingEnvActionWrapper, self).__init__(env)
    #     old_as = env.action_space
    #     num_actions = old_as.spaces[0].n
    #     self.action_space = gym.spaces.Tuple((
    #          old_as.spaces[0],  # 离散动作部分
    #          spaces.Box(np.float32(old_as.spaces[1].low), np.float32(old_as.spaces[1].high), dtype=np.float32),
    #          spaces.Box(np.float32(old_as.spaces[2].low), np.float32(old_as.spaces[2].high), dtype=np.float32),
    #
    #     ))

    def __init__(self, env):
        super(TaskOffloadingEnvActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(np.float32(old_as.spaces[1].spaces[i].low), np.float32(old_as.spaces[1].spaces[i].high),
                             dtype=np.float32)
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
