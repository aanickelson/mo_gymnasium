import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class MOHopperEnv(HopperEnv, EzPickle):
    """
    ## Description
    Multi-objective version of the HopperEnv environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/mujoco/hopper/) for more information.

    ## Reward Space
    The reward is 3-dimensional:
    - 0: Reward for going forward on the x-axis
    - 1: Reward for jumping high on the z-axis
    - 2: Control cost of the action
    If the cost_objective flag is set to False, the reward is 2-dimensional, and the cost is added to other objectives.
    """

    def __init__(self, cost_objective=True, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(self, cost_objective, **kwargs)
        self.cost_objetive = cost_objective
        self.reward_dim = 3 if cost_objective else 2
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,))
        self.st_bh_size = 11
        self.st_bh_idxs = [0, 11]

        self.rw_norm = 1000
        self.max_rw = [self.rw_norm] * self.reward_dim
        self.cumulative_reward = np.zeros((self.reward_dim,))
        self.cumulative_reward[2] = 2 * self.rw_norm
        self.fin_rw = np.zeros((self.reward_dim,))
        self.curr_ts = 0

    def calc_fin_rw(self):
        # Highest jump and longest jump
        self.fin_rw[0] = (self.cumulative_reward[0] / (self.rw_norm))
        self.fin_rw[1] = (self.cumulative_reward[1] / (self.rw_norm))

        if self.fin_rw[0] > 5 or self.fin_rw[1] > 5:
            print(self.fin_rw)
        # Energy used divided by total possible (2 units per time step)
        self.fin_rw[2] = (self.cumulative_reward[2] / (2 * self.rw_norm)) ** 6
        try:
            assert 0 < self.fin_rw[2] < 5
        except AssertionError:
            print(self.fin_rw[2], self.cumulative_reward[2])

    def reset_custom(self):
        self.fin_rw = np.zeros((self.reward_dim,))
        self.cumulative_reward[2] = 2 * self.rw_norm
        self.curr_ts = 0
        return self.reset()

    def step(self, action):
        self.curr_ts += 1
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt
        x_velocity = np.clip(x_velocity, 0, 3)
        # ctrl_cost = self.control_cost(action)

        # forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        # rewards = forward_reward + healthy_reward
        # costs = ctrl_cost

        observation = self._get_obs()
        # reward = rewards - costs
        terminated = self.terminated

        z = self.data.qpos[1]
        height = 10 * (z - self.init_qpos[1])
        energy_cost = np.sum(np.square(action))
        height = np.clip(height, 0, 3)

        if self.cost_objetive:
            vec_reward = np.array([x_velocity, height, -energy_cost], dtype=np.float32)
        else:
            vec_reward = np.array([x_velocity, height], dtype=np.float32)
            vec_reward -= self._ctrl_cost_weight * energy_cost

        vec_reward += healthy_reward

        # Save the furthest and highest jumps
        self.cumulative_reward += vec_reward
        self.calc_fin_rw()

        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "height_reward": height,
            "energy_reward": -energy_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, vec_reward, terminated, False, info
