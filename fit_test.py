import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.nonlinear = nn.Sigmoid()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.nonlinear,
            nn.Linear(hidden_size, hidden_size),
            self.nonlinear,
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# lr = 0.03
lr = 1e-2
env_size = 5
hidden_dim = 16
target_q_path = "result/optimal_qv_VI.npy"

batch_size = 16
epoches = 100000
state_dim = 2
action_num = 5

state_num = env_size**2
target_q = torch.from_numpy(np.load(target_q_path)).float()
states = torch.from_numpy(np.array([[y, x] for y in range(env_size) for x in range(env_size)])).float() 


def make_batch(N):
    select_idxs = np.random.choice(range(state_num), N)
    return states[select_idxs], target_q[select_idxs]


net = QNet(state_dim, hidden_dim, action_num)
q_table = torch.zeros((state_num, action_num))
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_bkp = []
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[epoches / 10 * i for i in range(1, 10)],
    gamma=0.8,
    last_epoch=-1,
)
for epoch in range(epoches):
    x, y = make_batch(batch_size)
    y_hat = net(x)
    loss = loss_func(y, y_hat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr_bkp.append(optimizer.param_groups[0]["lr"])

    if epoch % 100 == 0:
        with torch.no_grad():
            for i in range(env_size):
                for j in range(env_size):
                    q_table[i * env_size + j] = net(states[i * env_size + j])
        delta = (q_table - target_q).abs().mean()
        print(epoch, delta.item())
        if delta < 1e-2:
            break

np.save("trained_q_table.npy", q_table.numpy())
# torch.save(net.state_dict(), "optimal_Q_func.pth")

# 绘制学习率曲线
import matplotlib.pyplot as plt

plt.plot(lr_bkp)
plt.show()
