from gymnasium.envs.registration import register


register(
    id="mo-lunar-lander-v2",
    entry_point="mo_gymnasium.envs.lunar_lander.lunar_lander:MOLunarLander",
    max_episode_steps=500,
)

register(
    id="mo-lunar-lander-continuous-v2",
    entry_point="mo_gymnasium.envs.lunar_lander.lunar_lander:MOLunarLander",
    max_episode_steps=500,
    kwargs={"continuous": True},
)

register(
    id="mo-lunar-lander-continuous-new-rw-v2",
    entry_point="mo_gymnasium.envs.lunar_lander.lunar_lander_new_rw:MOLunarLander",
    max_episode_steps=100,
    kwargs={"continuous": True},
)
