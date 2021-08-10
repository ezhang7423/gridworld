import random
import math
from copy import deepcopy
MAX_ITERS = 20


class Node:
    def __init__(self, env, action, parent) -> None:
        self.env = env
        self.action = action
        self.num_samples = 0
        self.num_finished = 0
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
            return 0.5

        uct= (self.num_finished / self.num_samples) + (
            exploration
            * math.sqrt(math.log(self.parent.num_samples) / self.num_samples)
        )
        return uct


class MCTS:
    def __init__(self, env, simulations=1000, exploration=1) -> None:
        self.root = Node(deepcopy(env), None, None)
        self.root.expand_children()

        self.simulations = simulations
        self.exploration = exploration

    def find_move(self):
        for _ in range(self.simulations):
            node = self.select(self.root)
            node.expand_children()
            selected_node = self.uct(node.children)
            successful = self.rollout(selected_node)
            self.backup(selected_node, successful)

        best_choice = self.root.children[0]
        for child in self.root.children:
            if child.num_samples > best_choice.num_samples:
                best_choice = child

        # breakpoint()
        print(self.root.num_finished)
        print([c.num_samples for c in self.root.children])
        print([c.num_finished for c in self.root.children])
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
                return True
        return False

    def backup(self, node, successful):
        node.num_samples += 1
        node.num_finished += successful
        if node.parent:
            self.backup(node.parent, successful)

    def uct(self, list_of_nodes):
        return max(list_of_nodes, key=lambda node: node.uct(self.exploration))
