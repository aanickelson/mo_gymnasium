import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium.utils import EzPickle


class MOMountainCar(MountainCarEnv, EzPickle):
    """
    A multi-objective version of the MountainCar environment, where the goal is to reach the top of the mountain.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) for more information.

    ## Reward space:
    The reward space is a 3D vector containing the time penalty, and penalties for reversing and going forward.
    - time penalty: -1.0 for each time step
    - reverse penalty: -1.0 for each time step the action is 0 (reverse)
    - forward penalty: -1.0 for each time step the action is 2 (forward)
    """

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super().__init__(render_mode, goal_velocity)
        EzPickle.__init__(self, render_mode, goal_velocity)

        self.reward_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([-1, 0, 0]), shape=(3,), dtype=np.float32)
        self.reward_dim = 3

        self.st_bh_size = 2
        self.st_bh_idxs = [0, 2]
        self.rw_norm = 200
        self.max_rw = [self.rw_norm] * self.reward_dim
        self.cumulative_reward = np.zeros((self.reward_dim,))
        self.fin_rw = np.zeros((self.reward_dim,))

    def calc_fin_rw(self):
        # Calculate the percent of theoretical max reward we have achieved
        # 200 time steps (from init file), each time step adds up to -1 reward
        # 200 is the theoretical max, then subtract rewards from there
        # Normalize with 200 to get all rewards in [0, 1]
        self.fin_rw = (self.cumulative_reward + self.max_rw) / self.rw_norm

    def step(self, action: int):
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        # reward = -1.0
        reward = np.zeros(3, dtype=np.float32)
        reward[0] = 0.0 if terminated else -1.0  # time penalty
        reward[1] = 0.0 if action != 0 else -1.0  # reverse penalty
        reward[2] = 0.0 if action != 2 else -1.0  # forward penalty

        # Keeps track of NEGATIVE rewards so far
        self.cumulative_reward += reward
        # fin_rw turns them into positive rewards
        self.calc_fin_rw()

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
