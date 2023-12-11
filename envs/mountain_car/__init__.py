from gymnasium.envs.registration import register


register(
    id="mo-mountaincar-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car:MOMountainCar",
    max_episode_steps=200,
)

register(
    id="mo-mountaincar-new-rw-v0",
    entry_point="mo_gymnasium.envs.mountain_car.mountain_car_new_rw:MOMountainCar",
    max_episode_steps=200,
)
