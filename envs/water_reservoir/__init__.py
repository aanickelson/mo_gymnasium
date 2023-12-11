from gymnasium.envs.registration import register


register(id="water-reservoir-v0", entry_point="mo_gymnasium.envs.water_reservoir.dam_env:DamEnv")
register(id="water-reservoir-new-rw-v0", entry_point="mo_gymnasium.envs.water_reservoir.dam_env_new_rw:DamEnv")
