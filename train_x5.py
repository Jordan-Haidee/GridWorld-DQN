from datetime import datetime
from pathlib import Path
from DQN import DQNAgent
import utils

seed = 100
save_dir = Path("result") / f"DQN-{datetime.now().strftime(r'%Y-%m-%d-%H-%M-%S')}"
utils.start_tensorboard(save_dir)
utils.setup_seed(seed)

model = DQNAgent(
    "envs:GridWorld",
    env_size=5,
    seed=seed,
    hidden_dim=16,
    buffer_capicity=1000,
    buffer_init_ratio=0.999,
    lr=5e-2,
    lr_gamma=0.50,
    tau=0.005,
    update_interval=5,
    gamma=0.90,
    batch_size=64,
    save_dir=save_dir,
    total_steps=100000,
    optimal_q_table="result/optimal_qv_x5.npy",
)

model.train()
