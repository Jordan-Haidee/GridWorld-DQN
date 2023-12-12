import functools
from typing import Optional
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import gymnasium as gym


# 创建GridWorld环境
class GridWorld(gym.Env):
    map_reward = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -10, -10, -10, -10, 0, 0, 0, 0],
            [0, 0, -10, -10, -10, -10, 0, 0, 0, 0],
            [0, 0, 0, 0, -10, -10, 0, 0, 0, 0],
            [0, 0, 0, 0, -10, -10, 0, 0, 0, 0],
            [0, 0, -10, -10, 0, 0, -10, -10, 0, 0],
            [0, 0, -10, -10, 0, 1, -10, -10, 0, 0],
            [0, 0, -10, -10, 0, 0, 0, 0, 0, 0],
            [0, 0, -10, -10, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )
    goal = np.array([7, 5])  # 目标状态
    size = map_reward.shape[0]

    def __init__(self):
        self.state = np.array([0, 0], dtype=int)  # 初始状态
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.size, self.size]),
            dtype=int,
        )

    @classmethod
    @functools.lru_cache
    def _transitions(cls):
        trans = np.zeros((cls.size, cls.size, 5, 2), dtype=int)
        for i in range(cls.size):
            for j in range(cls.size):
                trans[i][j][0] = np.array([i, j]) if i == 0 else np.array([i - 1, j])
                trans[i][j][1] = np.array([i, j]) if i == cls.size - 1 else np.array([i + 1, j])
                trans[i][j][2] = np.array([i, j]) if j == 0 else np.array([i, j - 1])
                trans[i][j][3] = np.array([i, j]) if j == cls.size - 1 else np.array([i, j + 1])
                trans[i][j][4] = np.array([i, j])
        return trans

    @classmethod
    def _next_state(cls, state, action):
        i, j = state
        return cls._transitions()[i][j][action]

    @classmethod
    def _is_cross_border(cls, state, action):
        situations = [
            state[0] == 0 and action == 0,
            state[0] == cls.size - 1 and action == 1,
            state[1] == 0 and action == 2,
            state[1] == cls.size - 1 and action == 3,
        ]
        return True if any(situations) else False

    @classmethod
    def _reward(cls, state, action):
        i, j = cls._next_state(state, action)
        r1 = cls.map_reward[i][j]
        r2 = -1 if cls._is_cross_border(state, action) else 0
        return r1 + r2

    @property
    def transitions(self):
        return self._transitions()

    def next_state(self, action):
        i, j = self.state
        return self.transitions[i][j][action]

    def is_cross_border(self, action):
        """判断是否跨越边界"""
        return self._is_cross_border(self.state, action)

    def reward(self, action):
        """求取奖励"""
        return GridWorld._reward(self.state, action)

    def step(self, action):
        """切换状态并求取奖励"""
        r = self.reward(action)
        self.state = self.next_state(action)
        t1 = True if np.array_equal(self.state, self.goal) else False
        info = {}
        return self.state, r, t1, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """回到原点"""
        self.state = np.array([0, 0])
        info = {}
        return self.state, info

    @classmethod
    def index2coordinate(cls, idx):
        return np.array([idx // cls.size, idx % cls.size])

    @classmethod
    def coordinate2index(cls, coord):
        i, j = coord
        return i * cls.size + j

    def display(self, policy: Optional[np.ndarray] = None):
        # 设置颜色图
        _map = self.map_reward.copy()
        _map[_map == 0] = 10
        _map[_map == -10] = 20
        _map[_map == 1] = 30
        colors = ["#b4ff9a", "#ff4500", "#ffd700"]
        my_cmap = ListedColormap(colors, name="my_cmap")
        plt.imshow(_map, cmap=my_cmap)
        # 绘制策略
        if policy is not None:
            dxdy = [(0, -0.5), (0, 0.5), (-0.5, 0), (0.5, 0)]
            for i in range(self.size):
                print()
                for j in range(self.size):
                    if (a := policy[i][j]) < 4:
                        plt.arrow(j, i, *dxdy[a], width=0.01, length_includes_head=True, head_width=0.10, color="black")
                    else:
                        plt.scatter(j, i, marker="o", color="black",s=100)

        plt.xticks(np.arange(self.size) - 0.5)
        plt.yticks(np.arange(self.size) - 0.5)
        plt.grid(color="black", linestyle="-", linewidth=1)
        plt.show()
