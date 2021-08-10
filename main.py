import time
import mcts
import numpy as np
import gym
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

env = gym.make("MiniGrid-Empty-8x8-v0")
env = FullyObsWrapper(env)  # Get pixel observations

ORIENTATIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class Model_Env:
    def __init__(self, step_count=0, orientation=0, position=(0, 0)) -> None:

        self.orientation = orientation
        self.position = position
        self.step_count = step_count

    def step(self, action):
        if action == 0:
            self.orientation -= 1
            self.orientation %= 4
        if action == 1:
            self.orientation += 1
            self.orientation %= 4
        if action == 2:
            orientation = ORIENTATIONS[self.orientation]
            self.position = (
                Model_Env.calc_new_position(self.position[0], orientation[0]),
                Model_Env.calc_new_position(self.position[1], orientation[1]),
            )

    def copy(self):
        return Model_Env(self.step_count, self.orientation, self.position)

    def done(self):
        return self.position == (5, 5)

    @staticmethod
    def calc_new_position(old, new):
        return min(max(old + new, 0), 5)

    def reward(self):
        if self.position == (5, 5):
            return 1 - 0.9 * (self.step_count / 100)
        else:
            return 0

    def __str__(self) -> str:
        return f"Orientation: {self.orientation}, position: {self.position}"


# mct = mcts.MCTS(env)
env.render()
while True:
    print("Starting training loop")
    observation = env.reset()
    for t in range(100):
        env.render()

        pos = (env.agent_pos[0] - 1, env.agent_pos[1] - 1)
        mct = mcts.MCTS(Model_Env(orientation=env.agent_dir, position=pos))
        action = mct.find_move()
        # action = mct.find_move()
        print("Action: ", MiniGridEnv.Actions(action).name)
        observation, reward, done, info = env.step(action)
        # breakpoint()
        # print(env)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()