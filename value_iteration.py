import numpy as np
import gymnasium as gym


# 环境/参数设置
env = gym.make("envs:GridWorld", size=10)
gamma = 0.90
states = np.arange(env.unwrapped.size**2)
value_table = np.zeros_like(states, dtype=float)
value_table_bkp = value_table.copy()
# 执行值迭代算法
iter_num = 0
while True:
    iter_num += 1
    delta = 0
    for s in states:
        _s = env.unwrapped.index2coordinate(s)
        q_s = []
        for a in range(5):
            _ns = env.unwrapped._next_state(_s, a)
            ns = env.unwrapped.coordinate2index(_ns)
            q_s.append(env.unwrapped._reward(_s, a) + gamma * value_table[ns])
        value_table[s] = np.max(q_s)
    delta = np.abs(value_table_bkp - value_table).sum()
    value_table_bkp = np.copy(value_table)
    print(f"第{iter_num}次迭代, delta={delta}")
    if delta < 1e-3:
        break

# 提取最优策略
q_table = np.zeros((env.unwrapped.size**2, 5), dtype=float)
for s in states:
    _s = env.unwrapped.index2coordinate(s)
    q_s = []
    for a in range(5):
        _ns = env.unwrapped._next_state(_s, a)
        ns = env.unwrapped.coordinate2index(_ns)
        q_s.append(env.unwrapped._reward(_s, a) + gamma * value_table[ns])
    q_table[s] = np.array(q_s)
policy = q_table.argmax(axis=1).reshape(env.unwrapped.size, env.unwrapped.size)
env.unwrapped.display_optimal_policy(policy)

np.save(f"result/optimal_sv_x{env.unwrapped.size}.npy", value_table)
np.save(f"result/optimal_qv_x{env.unwrapped.size}.npy", q_table)
np.save(f"result/policy_table_x{env.unwrapped.size}.npy", policy)
