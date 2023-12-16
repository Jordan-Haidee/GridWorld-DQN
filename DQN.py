from datetime import datetime
from pathlib import Path
import random
from collections import Counter, deque
from typing import Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from torch.utils import tensorboard as tb
from tqdm import trange


class ReplayBuffer:
    def __init__(self, capicity: int) -> None:
        self.capicity = capicity
        self.buffer = deque(maxlen=self.capicity)

    @property
    def size(self):
        return len(self.buffer)

    def is_full(self):
        return True if self.size >= self.capicity else False

    def push(self, s, a, r, ns, d):
        if self.is_full():
            self.buffer.popleft()
        self.buffer.append((s, a, r, ns, d))

    def sample(self, N: int):
        """采样数据并打包"""
        assert N <= self.size, "batch is too big"
        samples = random.sample(self.buffer, N)
        s, a, r, ns, d = zip(*samples)
        return (
            torch.from_numpy(np.vstack(s)).float(),
            torch.from_numpy(np.vstack(a)).type(torch.int64),
            torch.from_numpy(np.vstack(r)).float(),
            torch.from_numpy(np.vstack(ns)).float(),
            torch.from_numpy(np.vstack(d)).float(),
        )

    def count_distribution(self):
        """统计状态分布"""
        counter = Counter()
        idxs = []
        for trans in self.buffer:
            idxs.append((tuple(trans[0]), trans[1]))
        counter.update(idxs)
        return counter


class QNet(nn.Module):
    """Q网络, 用于替代Q表格"""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.nonlinear = nn.Sigmoid()

    def forward(self, state):
        x = self.fc1(state)
        x = self.nonlinear(x)
        x = self.fc2(x)
        x = self.nonlinear(x)
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(
        self,
        env: str,
        env_size: int = 10,
        seed: Optional[int] = None,
        hidden_dim: int = 100,
        buffer_capicity: int = 2000,
        buffer_init_ratio: float = 0.30,
        lr: float = 1e-4,
        lr_gamma: float = 0.80,
        lr_miletones: Sequence[int] = None,
        tau: float = 0.005,
        update_interval: int = 10,
        gamma: float = 0.90,
        batch_size: int = 64,
        save_dir: Union[str, Path] = Path("result") / f"DQN-{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}",
        total_steps: int = 100000,
        optimal_q_table: str = "result/optimal_qv_x10.npy",
    ) -> None:
        self.env = gym.make(env, size=env_size, env_type="train")
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.q_net = QNet(state_dim, hidden_dim, action_dim)
        self.q_target = QNet(state_dim, hidden_dim, action_dim)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr)
        self.total_steps = total_steps
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=lr_miletones,
            gamma=lr_gamma,
            last_epoch=-1,
        )
        self.replay_buffer = ReplayBuffer(buffer_capicity)
        self.buffer_init_ratio = buffer_init_ratio
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.update_interval = update_interval
        self.optimal_q_table = torch.from_numpy(np.load(optimal_q_table)).float()
        self.loss_fn = nn.MSELoss()

    @property
    def epsilon(self):
        p = self._step / self.total_steps
        return max(np.exp(-4 * p), 0.50)

    @torch.no_grad()
    def choose_action(self, s: np.ndarray, epsilon: float = 0.0) -> int:
        """选择动作"""
        if np.random.uniform() < epsilon:
            a = self.env.action_space.sample()
        else:
            s_ = torch.from_numpy(s).float()
            a = self.q_net(s_).argmax().item()
        return a

    def collect_exp_before_train(self):
        """在训练前收集经验"""
        N = self.buffer_init_ratio * self.replay_buffer.capicity
        s, _ = self.env.reset()
        while self.replay_buffer.size < N:
            _s = self.env.unwrapped.normalize_state(s)
            a = self.choose_action(_s, 1.0)
            ns, r, t1, t2, info = self.env.step(a)
            _s = self.env.unwrapped.normalize_state(s)
            _ns = self.env.unwrapped.normalize_state(ns)
            self.replay_buffer.push(_s, a, r, _ns, t1)
            s = self.env.reset()[0] if t1 or t2 else ns

    def soft_sync_target(self):
        for p, p_ in zip(self.q_net.parameters(), self.q_target.parameters()):
            p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))

    def hard_sync_target(self):
        self.q_target.load_state_dict(self.q_net.state_dict())

    def train_one_batch(self):
        """训练一个mini-batch"""
        s, a, r, ns, d = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            td_target = r + self.gamma * self.q_target(ns).max(dim=1, keepdim=True)[0]
        loss = self.loss_fn(td_target, self.q_net(s).gather(1, a))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.detach()

    @property
    def lr(self):
        """获得实时学习率"""
        return self.optimizer.param_groups[0]["lr"]

    def log_info_per_batch(self, loss):
        """记录日志"""
        self.logger.add_scalar("Loss/td_loss", loss, self._step)
        self.logger.add_scalar("Buffer/buffer_size", self.replay_buffer.size, self._step)
        self.logger.add_scalar("Epsilon/epsilon", self.epsilon, self._step)
        self.logger.add_scalar("lr", self.lr, self._step)
        if self._step % 100 == 0:
            l1, l2 = self.compute_absolute_loss()
            self.logger.add_scalar("Loss/absolute_q_loss/mean", l1, self._step)
            self.logger.add_scalar("Loss/absolute_q_loss/max", l2, self._step)

    @torch.no_grad()
    def compute_absolute_loss(self):
        q_table = torch.zeros_like(self.optimal_q_table).float()
        size = self.env.unwrapped.size
        for i in range(size):
            for j in range(size):
                s = torch.tensor([i, j]).float()
                _s = self.env.unwrapped.normalize_state(s)
                q_table[i * size + j] = self.q_net(_s)
        loss = (q_table - self.optimal_q_table).abs()
        return loss.mean(), loss.max()

    def save(self):
        """保存模型"""
        params = {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
        }
        torch.save(params, self.save_dir / "latest.pt")

    def train(self):
        """进行训练"""
        self.logger = tb.SummaryWriter(log_dir=self.save_dir / "train_log")
        self.save_dir.mkdir(exist_ok=True)
        self.collect_exp_before_train()
        s, _ = self.env.reset()
        for _n in trange(self.total_steps):
            self._step = _n
            _s = self.env.unwrapped.normalize_state(s)
            a = self.choose_action(_s, 1.0)
            ns, r, t1, t2, info = self.env.step(a)
            _ns = self.env.unwrapped.normalize_state(ns)
            # self.replay_buffer.push(_s, a, r, _ns, t1)
            loss = self.train_one_batch()
            if t1 or t2:
                s, _ = self.env.reset()
            else:
                s = ns
            self.log_info_per_batch(loss)
            if self._step % self.update_interval == 0:
                self.hard_sync_target()
            # self.soft_sync_target()
        self.save()
        self.test(self.save_dir / "latest.pt")

    @torch.no_grad()
    def test(self, checkpoint: Optional[Union[str, Path]] = None):
        """测试并打印每一个状态的最优动作"""
        if checkpoint is not None:
            params = torch.load(checkpoint)
            self.q_net.load_state_dict(params.get("q_net"))
        s, _ = self.env.reset()
        size = self.env.unwrapped.size
        policy = np.zeros((size, size), dtype=int)
        q_table = np.zeros((size, size, self.env.action_space.n), dtype=float)
        for i in range(self.env.unwrapped.size):
            for j in range(self.env.unwrapped.size):
                s = np.array([i, j], dtype=float)
                _s = self.env.unwrapped.normalize_state(s)
                _s = torch.from_numpy(_s).float()
                q_table[i][j] = self.q_net(_s).numpy()
                policy[i][j] = q_table[i][j].argmax()
        self.env.unwrapped.display(policy)
