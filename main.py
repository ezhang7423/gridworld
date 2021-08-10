import mcts
import numpy as np
import gym
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

env = gym.make("MiniGrid-Empty-8x8-v0")
env = FullyObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)  # Get rid of the 'mission' field

ORIENTATIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class Model_Env:
    def __init__(
        self, state=None, step_count=0, orientation=0, position=(0, 0)
    ) -> None:
        if state is None:
            self.state = np.zeros((8, 8))
        else:
            self.state = state

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
        return Model_Env(
            np.copy(self.state), self.step_count, self.orientation, self.position
        )

    def done(self):
        return self.position == (7, 7)
    @staticmethod
    def calc_new_position(old, new):
        return min(max(old + new, 0), 7)

    def reward(self):
        if self.position == (7, 7):
            return 1 - 0.9 * (self.step_count / 100)
        else:
            return 0


for i_episode in range(20):
    observation = env.reset()
    # mct = mcts.MCTS(env)
    print("Starting training loop")
    mct = mcts.MCTS(Model_Env())

    for t in range(100):
        env.render()
        action, mct = mct.find_move()
        # action = mct.find_move()
        print("Action: ", MiniGridEnv.Actions(action).name)
        observation, reward, done, info = env.step(action)
        # breakpoint()
        print(env)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
