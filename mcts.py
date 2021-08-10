from copy import deepcopy
import random
import math

MAX_ITERS = 100


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
            for action in range(7):
                child_env = self.env.copy()
                child_env.step(action)
                self.children.append(Node(child_env, action, self))
            self.is_leaf = False

    def uct(self, exploration):
        if self.num_samples == 0:  # bias towards exploring new states
            return 10

        uct = (self.total_reward / self.num_samples) + (
            exploration
            * math.sqrt(math.log(self.parent.num_samples) / self.num_samples)
        )
        # print(uct)
        return uct


class MCTS:
    def __init__(self, env, simulations=3000, exploration=3) -> None:
        self.root = Node(env.copy(), None, None)
        self.root.expand_children()

        self.simulations = simulations
        self.exploration = exploration

    def find_move(self):
        for _ in range(self.simulations):
            # if _ % 100 == 0:
            #     print(_)
            #     print(self.root.total_reward)
            #     print([(c.num_samples, c.total_reward) for c in self.root.children])
            #     print()
            node = self.select(self.root)
            node.expand_children()
            selected_node = self.uct(node.children)
            reward = self.rollout(selected_node)
            self.backup(selected_node, reward)

        best_choice = self.root.children[0]
        for child in self.root.children:
            if child.num_samples > best_choice.num_samples:
                best_choice = child

        # new_root = deepcopy(best_choice)
        # new_root.depth = 0
        # new_root.parent = None
        # new_root.action = None
        # self.root = new_root
        # return best_choice.action, self
        return best_choice.action

    def select(self, node):
        # if node.depth > MAX_ITERS:
        #     return node.parent

        if node.is_leaf:
            return node
        return self.select(self.uct(node.children))

    def rollout(self, node):
        # total_reward = 0
        tmp_env = node.env.copy()
        for i in range(MAX_ITERS):

            action = random.randint(0, 6)
            # need to create a new env each time
            node.env.step(action)
            # total_reward += reward
            if node.env.done():
                return node.env.reward()

        return 0

    def backup(self, node, reward):
        node.num_samples += 1
        node.total_reward += reward
        if node.parent:
            self.backup(node.parent, reward)

    def uct(self, list_of_nodes):
        best = [list_of_nodes[0]]
        best_uct = list_of_nodes[0].uct(self.exploration)
        for n in list_of_nodes:
            contender_uct = n.uct(self.exploration)
            if contender_uct > best_uct:
                best = [n]
                best_uct = contender_uct
            elif contender_uct == best_uct:
                best.append(n)

        return random.choice(best)
