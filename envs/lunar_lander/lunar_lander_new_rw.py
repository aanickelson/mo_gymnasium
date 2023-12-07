import math

import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.lunar_lander import (
    FPS,
    LEG_DOWN,
    MAIN_ENGINE_POWER,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    VIEWPORT_H,
    VIEWPORT_W,
    LunarLander,
)


class MOLunarLander(LunarLander):  # no need for EzPickle, it's already in LunarLander
    """
    ## Description
    Multi-objective version of the LunarLander environment.

    See [Gymnasium's env](https://gymnasium.farama.org/environments/box2d/lunar_lander/) for more information.

    ## Reward Space
    The reward is 4-dimensional:
    - 0: -100 if crash, +100 if lands successfully
    - 1: Shaping reward
    - 2: Fuel cost (main engine)
    - 3: Fuel cost (side engine)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("Running Lunar Lander with custom rewards")
        # Set up so we're looking to maximize each one at 1
        # Rewards are:
        # smallest distance to goal (small distance better)
        # vertical velocity as it approaches the ground (small velocity better)
        # angle as it approaches the ground (small angle better)
        # average fuel used
        self.reward_space = spaces.Box(
            low=np.array([0., 0., 0., 0.]),
            high=np.array([1., 1., 1., 1.]),
            shape=(4,),
            dtype=np.float32,
        )
        self.reward_dim = 4
        self.goal = np.zeros((2,))
        self.max_dist = self.calc_max_dist()
        self.st_bh_size = 6
        self.st_bh_idxs = [0, 6]

        self.fin_rw = np.zeros((4,))
        self.tot_fuel = 0
        self.ts = 0

    def reset_custom(self):
        self.fin_rw = np.zeros((4,))
        self.tot_fuel = 0
        self.ts = 0

    def calc_max_dist(self):
        highs = self.observation_space.high[0:2]
        return np.linalg.norm(self.goal - highs)

    def step(self, action):
        st, vec_reward, terminated, truncated, info = self.original_code(action)

        # Distance to goal
        # Euclidean distance between x,y and [0, 0]
        pos = np.array(st[:2])
        dist = np.linalg.norm(pos - self.goal)
        # Interpolate to [0, 1] where 1 is reached goal
        new_dist = np.interp(np.abs(dist), [0, self.max_dist], [1, 0])

        # Vertical velocity when approaching the ground
        lin_vel = np.array(st[2])
        # Want it to come in slow for a soft landing - neither highly negative or positive
        vert_vel = np.interp(np.abs(lin_vel), [0, self.observation_space.high[2]], [1, 0])

        # Angle when approaching the ground
        ang = np.array(st[4])
        # It doesn't actually get limited to -pi, pi, so we must do that ourselves
        new_ang = abs(ang) % (2 * math.pi)
        if new_ang > math.pi:
            new_ang = (2 * math.pi) - new_ang
        # Want it to be as close to 0 as possible
        ang_interp = np.interp(np.abs(new_ang), [0, self.observation_space.high[4]], [1, 0])

        # Save the closest distance to the goal, velocity, and angle at that time
        if new_dist > self.fin_rw[0]:
            self.fin_rw[0] = new_dist
            self.fin_rw[1] = vert_vel
            self.fin_rw[2] = ang_interp

        # Fuel used at this time step
        fuel = np.sum(vec_reward[3:])
        # Both fuel should always be in [0.5, 1] so their sum should be in [1, 2]
        # Got this from analyzing the original code below, specifically the m_power and s_power clip lines
        if fuel == 0:
            fuel_interp = 0
        else:
            fuel_interp = np.interp(fuel, [1, 2], [1, 0.01])
        self.tot_fuel += fuel_interp
        self.ts += 1
        # Average fuel used over time
        self.fin_rw[3] = self.tot_fuel / self.ts

        return st, vec_reward, terminated, truncated, info

    def original_code(self, action):

        assert self.lander is not None

        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = math.tanh(math.sin(0.02 * self.wind_idx) + (math.sin(math.pi * 0.01 * self.wind_idx))) * self.wind_power
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(math.sin(0.02 * self.torque_idx) + (math.sin(math.pi * 0.01 * self.torque_idx))) * (
                self.turbulence_power
            )
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        vector_reward = np.zeros(4, dtype=np.float32)
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
            vector_reward[1] = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heuristic landing
        vector_reward[2] = -m_power
        reward -= s_power * 0.03
        vector_reward[3] = -s_power

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            vector_reward[0] = -100
        if not self.lander.awake:
            terminated = True
            reward = +100
            vector_reward[0] = +100

        if self.render_mode == "human":
            self.render()

        return np.array(state, dtype=np.float32), vector_reward, terminated, False, {"original_reward": reward}
