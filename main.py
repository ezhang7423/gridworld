import mcts
import gym
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

env = gym.make("MiniGrid-Empty-8x8-v0")
env = RGBImgPartialObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)  # Get rid of the 'mission' field

for i_episode in range(20):
    observation = env.reset()
    # mct = mcts.MCTS(env)
    print("Starting training loop")
    for t in range(100):
        mct = mcts.MCTS(env)
        # env.render()
        # action, mct = mct.find_move()
        action = mct.find_move()
        print("Action: ", MiniGridEnv.Actions(action).name)
        observation, reward, done, info = env.step(action)
        # breakpoint()
        print(env)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
