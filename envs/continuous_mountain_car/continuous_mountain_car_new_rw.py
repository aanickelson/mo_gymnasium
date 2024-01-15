import math
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)
from gymnasium.utils import EzPickle


class MOContinuousMountainCar(Continuous_MountainCarEnv, EzPickle):
    """
    A continuous version of the MountainCar environment, where the goal is to reach the top of the mountain.

    See [source](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) for more information.

    ## Reward space:
    The reward space is a 2D vector containing the time penalty and the fuel reward.
    - time penalty: -1.0 for each time step
    - fuel reward: -||action||^2 , i.e. the negative of the norm of the action vector
    """

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        super().__init__(render_mode, goal_velocity)
        EzPickle.__init__(self, render_mode, goal_velocity)

        self.reward_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([0.0, 0.0]), dtype=np.float32)
        self.reward_dim = 2
        self.st_bh_size = 2
        self.st_bh_idxs = [0, 1]
        self.rw_norm = 200
        self.curr_ts = 0
        # Maximum distance it can get away from the goal
        # We are minimizing the closest it gets to the goal. It starts somewhere between [-0.6, -0.4]
        # So the biggest minimum distance it can have is the goal position to -0.6
        self.dist_norm = self.goal_position - (-0.6)
        self.max_rw = [self.rw_norm] * self.reward_dim
        self.cumulative_reward = np.zeros((self.reward_dim,))
        self.fin_rw = np.zeros((self.reward_dim,))

    def calc_fin_rw(self):
        # Calculate the percent of theoretical max reward we have achieved
        # 200 time steps (from init file), each time step adds up to -1 reward
        # 200 is the theoretical max, then subtract rewards from there
        # Normalize with 999 to get all rewards in [0, 1]
        # cubed so there's a bigger difference as you approach 1
        self.fin_rw[1] = ((self.cumulative_reward[1] + self.max_rw[1]) / self.rw_norm)
        time_rw = (self.rw_norm - self.curr_ts) / self.rw_norm
        self.fin_rw[0] = (self.cumulative_reward[0] + time_rw) / 2

    def reset_custom(self):
        self.cumulative_reward = np.zeros((self.reward_dim,))
        self.fin_rw = np.zeros((self.reward_dim,))
        self.curr_ts = 0
        return self.reset()

    def step(self, action: np.ndarray):
        # Essentially a copy paste from original env, except the rewards

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        truncated = bool(self.curr_ts >= self.rw_norm)

        # Minimizes distance to goal by turning it into a maximization
        # If I reach the goal, the reward is 1
        # The further I am from the goal, the closer to 0
        dist_to_goal = 1 - (abs(position - self.goal_position) / self.dist_norm)

        if dist_to_goal > self.cumulative_reward[0]:
            self.cumulative_reward[0] = dist_to_goal

        reward = np.zeros(2, dtype=np.float32)
        # Time reward is negative at all timesteps except when reaching the goal
        # if terminated:
        #     reward[0] = 10.0
        # else:
        #     reward[0] = -1.0

        # Actions cost fuel, which we want to optimize too
        reward[1] = -math.pow(action[0], 2)

        # Keeps track of NEGATIVE rewards so far
        self.cumulative_reward[1] += reward[1]
        # fin_rw turns them into positive rewards
        self.calc_fin_rw()

        self.state = np.array([position, velocity], dtype=np.float32)

        self.curr_ts += 1

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, truncated, {}
