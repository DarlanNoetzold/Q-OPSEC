import random
from typing import Dict

class EpsilonGreedyPolicy:
    def __init__(self, actions: Dict[int, str], epsilon: float = 0.2):
        self.actions = actions
        self.epsilon = epsilon

    def select_action(self, state: str, q_table: Dict[str, Dict[int, float]]) -> int:
        if state not in q_table or random.random() < self.epsilon:
            return random.choice(list(self.actions.keys()))
        return max(q_table[state], key=q_table[state].get)