import gym
from gym import spaces
import numpy as np
import pygame
import sys
from scipy.integrate import odeint
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        print(f"Step: {self.num_timesteps}")
        return True


class MotorRotationEnv(gym.Env):
    def __init__(self):
        super(MotorRotationEnv, self).__init__()
        self.dt = 0.01
        self.target_angle = None

        # torque
        self.action_space = spaces.Box(low=np.array([-20]), high=np.array([20]), shape=(1,), dtype=np.float32)

        # [angle, dot_angle]
        self.observation_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([2*np.pi, np.pi]), shape=(2,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption('Motor Rotation')
        self.clock = pygame.time.Clock()

        self.seed = np.random.seed

    def reset(self):
        self.target_angle = np.random.uniform(0, 2*np.pi)
        print(f"target_angle: {self.target_angle}")
        self._state = np.array([0, 0])
        return self._state

    def _dynamics(self, state, t, u):
        dxdt = np.array([
            state[1], u
        ])
        return dxdt.tolist()

    def step(self, action):
        u = np.clip(action[0], -20, 20)
        new_state = odeint(self._dynamics, self._state, [0, self.dt], (u,))
        self._state = np.array(new_state[-1])

        error = np.abs(self.target_angle - self._state[0])
        reward = - error**2

        done = False
        if error < 0.01:
            done = True
            reward = 10
            
        return self._state, reward, done, {}

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(100, 100, 200, 200), 2)
        pygame.draw.line(self.screen, (255, 0, 0), (200, 200), (200 + 100 * np.cos(self._state[0]),
                                                               200 - 100 * np.sin(self._state[0])), 3)

        pygame.draw.line(self.screen, (0, 255, 0), (200, 200), (200 + 100 * np.cos(self.target_angle),
                                                               200 - 100 * np.sin(self.target_angle)), 3)

        pygame.display.flip()
        self.clock.tick(60)


gym.register('MotorRotation-v0', entry_point='r_all:MotorRotationEnv')

env = gym.make('MotorRotation-v0')

model = SAC("MlpPolicy", env, verbose=1)
callback = ProgressCallback()
model.learn(total_timesteps=int(1e3), callback=callback)

# model.save("sac_motor_rotation")
# model = SAC.load("sac_motor_rotation")

print('evolution...')
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    
    if done:
        print("done")
        obs = env.reset()
    

# test odeint()
# obj = MotorRotationEnv()
# obj.reset()
# while True:
#     state = obj.step([20])
#     obj.render()