from DQN import DQNAgent

model = DQNAgent(
    "envs:GridWorld",
    env_size=10,
    hidden_dim=80,
    buffer_capicity=20000,
    buffer_init_ratio=0.999,
    lr=5e-2,
    tau=0.005,
    update_interval=5,
    gamma=0.90,
    batch_size=256,
    total_steps=100000,
    lr_gamma=0.80,
    optimal_q_table="result/optimal_qv_x10.npy",
)

model.test("result/DQN-best_x10/latest.pt")

