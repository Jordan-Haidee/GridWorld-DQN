from DQN import DQNAgent

model = DQNAgent(
    "envs:GridWorld",
    env_size=5,
    hidden_dim=16,
    buffer_capicity=1000,
    buffer_init_ratio=0.999,
    lr=5e-2,
    lr_gamma=0.50,
    tau=0.005,
    update_interval=5,
    gamma=0.90,
    batch_size=64,
    total_steps=100000,
    optimal_q_table="result/optimal_qv_x5.npy",
)
model.test("result/DQN-best_x5/latest.pt")
