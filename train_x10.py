from datetime import datetime
from pathlib import Path
from DQN import DQNAgent
import utils

save_dir = Path("result") / f"DQN-{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}"
utils.start_tensorboard(save_dir)

model = DQNAgent(
    "envs:GridWorld",
    env_size=10,
    hidden_dim=80,
    buffer_capicity=50000,
    buffer_init_ratio=0.999,
    total_steps=200000,
    lr=5e-2,
    lr_gamma=0.20,
    lr_miletones=[200000 // 3 * i for i in range(1, 3)],
    tau=0.005,
    update_interval=5,
    gamma=0.90,
    batch_size=512,
    save_dir=save_dir,
    optimal_q_table="result/optimal_qv_x10.npy",
)

model.train()
