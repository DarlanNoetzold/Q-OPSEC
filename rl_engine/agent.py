from typing import Dict
from environment import ACTIONS

class RLAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.alpha = alpha
        self.gamma = gamma

    def update(self, state: str, action: int, reward: float, next_state: str):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in ACTIONS}

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value