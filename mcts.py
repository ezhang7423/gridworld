import random
import math
from copy import deepcopy
MAX_ITERS = 20


class Node:
    def __init__(self, env, action, parent) -> None:
        self.env = env
        self.action = action
        self.num_samples = 0
        self.total_reward = 0
        self.depth = 0
        self.parent = parent
        if self.parent:
            self.depth = self.parent.depth + 1
        self.children = []
        self.is_leaf = True

    def expand_children(self):
        if self.is_leaf:
            for action in range(self.env.action_space.n):
                child_env = deepcopy(self.env)
                child_env.step(action)
                self.children.append(Node(child_env, action, self))
            self.is_leaf = False

    def uct(self, exploration):
        if self.num_samples == 0:  # bias towards exploring new states
            return math.inf

        uct= (self.total_reward / self.num_samples) + (
            exploration
            * math.sqrt(math.log(self.parent.num_samples) / self.num_samples)
        )
        return uct


class MCTS:
    def __init__(self, env, simulations=10000, exploration=.9) -> None:
        self.root = Node(deepcopy(env), None, None)
        self.root.expand_children()

        self.simulations = simulations
        self.exploration = exploration

    def find_move(self):
        for _ in range(self.simulations):
            if _ % 100 == 0 or _ == self.simulations - 1:                
                print(_)
                print(self.root.total_reward)
                print([(c.num_samples, c.total_reward) for c in self.root.children])
                print()
            node = self.select(self.root)
            node.expand_children()
            selected_node = self.uct(node.children)
            reward = self.rollout(selected_node)
            self.backup(selected_node, reward)

        best_choice = self.root.children[0]
        for child in self.root.children:
            if child.num_samples > best_choice.num_samples:
                best_choice = child

        # breakpoint()

        # return best_choice.action, MCTS(best_choice.env)
        return best_choice.action

    def select(self, node):
        if node.depth > MAX_ITERS:
            return node.parent

        if node.is_leaf:
            return node
        return self.select(self.uct(node.children))

    def rollout(self, node):
        # total_reward = 0
        tmp_env = deepcopy(node.env)
        for i in range(MAX_ITERS):
            
            action = tmp_env.action_space.sample()
            # need to create a new env each time
            _, reward, done, _ = node.env.step(action)
            # total_reward += reward
            if done:
                return reward
        return 0

    def backup(self, node, reward):
        node.num_samples += 1
        node.total_reward += reward
        if node.parent:
            self.backup(node.parent, reward)

    def uct(self, list_of_nodes):
        return max(list_of_nodes, key=lambda node: node.uct(self.exploration))
