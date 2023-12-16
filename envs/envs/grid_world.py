import functools
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import pytoml
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


# 创建GridWorld环境
class GridWorld(gym.Env):
    def __init__(
        self,
        size: Literal[5, 10] = 10,
        env_type: Literal["train", "test"] = "train",
        seed: Optional[int] = None,
    ):
        with open(Path(__file__).parent / "default.toml") as f:
            config = pytoml.load(f)
        self.env_type = env_type
        self.size = size
        self.map_reward = np.array(config.get(f"map_reward_x{self.size}"), dtype=float)
        self.goal = config.get(f"goal_x{self.size}")
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.size, self.size]),
            dtype=int,
        )
        if seed is not None:
            self.action_space.seed(seed)

    @property
    @functools.lru_cache
    def transitions(self):
        trans = np.zeros((self.size, self.size, self.action_space.n, 2), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                trans[i][j][0] = np.array([i, j]) if i == 0 else np.array([i - 1, j])
                trans[i][j][1] = np.array([i, j]) if i == self.size - 1 else np.array([i + 1, j])
                trans[i][j][2] = np.array([i, j]) if j == 0 else np.array([i, j - 1])
                trans[i][j][3] = np.array([i, j]) if j == self.size - 1 else np.array([i, j + 1])
                trans[i][j][4] = np.array([i, j])
        return trans

    def _next_state(self, state, action):
        i, j = state
        return self.transitions[i][j][action]

    def _is_cross_border(self, state, action):
        situations = [
            state[0] == 0 and action == 0,
            state[0] == self.size - 1 and action == 1,
            state[1] == 0 and action == 2,
            state[1] == self.size - 1 and action == 3,
        ]
        return True if any(situations) else False

    def _reward(self, state, action):
        i, j = self._next_state(state, action)
        r1 = self.map_reward[i][j]
        r2 = -1 if self._is_cross_border(state, action) else 0
        return r1 + r2

    def next_state(self, action):
        i, j = self.state
        return self.transitions[i][j][action]

    def is_cross_border(self, action):
        """判断是否跨越边界"""
        return self._is_cross_border(self.state, action)

    def reward(self, action):
        """求取奖励"""
        return self._reward(self.state, action)

    def step(self, action):
        """切换状态并求取奖励"""
        r = self.reward(action)
        self.state = self.next_state(action)
        if self.env_type == "test":
            t1 = True if np.array_equal(self.state, self.goal) else False
        else:
            t1 = False
        info = {}
        return self.state, r, t1, False, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """回到原点"""
        if self.env_type == "test":
            self.state = np.array([0, 0])
        else:
            self.state = np.random.randint(0, self.size, size=(2,))
        info = {}
        return self.state, info

    def index2coordinate(self, idx):
        return np.array([idx // self.size, idx % self.size])

    def coordinate2index(self, coord):
        i, j = coord
        return i * self.size + j

    def normalize_state(self, s: np.ndarray):
        return s / 1.0
        # return s / self.size

    def normalize_reward(self, r: float):
        return r / 10.0

    def display(self, policy: Optional[np.ndarray] = None):
        # 设置颜色图
        _map = self.map_reward.copy()
        _map[_map == 0] = 10
        _map[_map == -10] = 20
        _map[_map == 1] = 30
        colors = ["#b4ff9a", "#ff4500", "#ffd700"]
        my_cmap = ListedColormap(colors, name="my_cmap")
        fig, ax = plt.subplots()
        ax.imshow(_map, cmap=my_cmap)
        # 绘制策略
        if policy is not None:
            dxdy = [(0, -0.5), (0, 0.5), (-0.5, 0), (0.5, 0)]
            for i in range(self.size):
                print()
                for j in range(self.size):
                    if (a := policy[i][j]) < 4:
                        ax.arrow(j, i, *dxdy[a], width=0.01, length_includes_head=True, head_width=0.10, color="black")
                    else:
                        ax.scatter(j, i, marker="o", color="black", s=100)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        return ax
