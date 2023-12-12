from gymnasium.envs.registration import register

register(
    id="GridWorld",
    entry_point="envs.envs:GridWorld",
    max_episode_steps=500,
)
