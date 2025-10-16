import random
import numpy as np
from typing import Dict, List
from environment import CryptoAlgorithm


class BasePolicy:
    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        raise NotImplementedError


class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon: float = 0.2, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Args:
            epsilon: Initial exploration rate
            epsilon_decay: Decay factor per episode
            epsilon_min: Minimum exploration rate
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon

    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        if not q_values:
            return random.choice(valid_actions)

        valid_q = {a: q_values.get(a, 0.0) for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self):
        self.epsilon = self.initial_epsilon


class BoltzmannPolicy(BasePolicy):
    def __init__(self, temperature: float = 1.0, temperature_decay: float = 0.995,
                 temperature_min: float = 0.1):
        """
        Args:
            temperature: Initial temperature (higher = more exploration)
            temperature_decay: Decay factor per episode
            temperature_min: Minimum temperature
        """
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min
        self.initial_temperature = temperature

    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        if not q_values:
            return random.choice(valid_actions)

        valid_q = np.array([q_values.get(a, 0.0) for a in valid_actions])

        # Compute Boltzmann probabilities
        # P(a) ∝ exp(Q(s,a) / T)
        exp_q = np.exp(valid_q / self.temperature)
        probs = exp_q / np.sum(exp_q)

        # Sample action
        action_idx = np.random.choice(len(valid_actions), p=probs)
        return valid_actions[action_idx]

    def decay(self):
        self.temperature = max(self.temperature_min,
                               self.temperature * self.temperature_decay)

    def reset(self):
        self.temperature = self.initial_temperature


class UCBPolicy(BasePolicy):
    def __init__(self, c: float = 2.0):
        """
        Args:
            c: Exploration constant (higher = more exploration)
        """
        self.c = c
        self.action_counts = {}
        self.total_count = 0

    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        state_key = str(state)
        if state_key not in self.action_counts:
            self.action_counts[state_key] = {a: 0 for a in valid_actions}

        for a in valid_actions:
            if a not in self.action_counts[state_key]:
                self.action_counts[state_key][a] = 0

        for a in valid_actions:
            if self.action_counts[state_key][a] == 0:
                self.action_counts[state_key][a] += 1
                self.total_count += 1
                return a

        ucb_values = {}
        for a in valid_actions:
            q_value = q_values.get(a, 0.0)
            count = self.action_counts[state_key][a]

            # UCB formula: Q(s,a) + c * sqrt(ln(N) / n(s,a))
            exploration_bonus = self.c * np.sqrt(np.log(self.total_count + 1) / count)
            ucb_values[a] = q_value + exploration_bonus

        selected_action = max(ucb_values, key=ucb_values.get)
        self.action_counts[state_key][selected_action] += 1
        self.total_count += 1

        return selected_action

    def reset(self):
        self.action_counts = {}
        self.total_count = 0


class AdaptivePolicy(BasePolicy):
    def __init__(self):
        self.epsilon_greedy = EpsilonGreedyPolicy(epsilon=0.3)
        self.boltzmann = BoltzmannPolicy(temperature=1.5)
        self.ucb = UCBPolicy(c=2.0)

        self.current_strategy = "epsilon_greedy"
        self.episode_count = 0
        self.performance_history = []

    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        if self.current_strategy == "epsilon_greedy":
            return self.epsilon_greedy.select_action(state, q_values, valid_actions)
        elif self.current_strategy == "boltzmann":
            return self.boltzmann.select_action(state, q_values, valid_actions)
        elif self.current_strategy == "ucb":
            return self.ucb.select_action(state, q_values, valid_actions)
        else:
            return random.choice(valid_actions) if valid_actions else 0

    def update_strategy(self, episode_reward: float):
        self.episode_count += 1
        self.performance_history.append(episode_reward)

        if self.episode_count % 100 == 0 and len(self.performance_history) >= 100:
            recent_performance = np.mean(self.performance_history[-100:])

            if recent_performance < 0:
                self.current_strategy = "ucb"
            elif recent_performance > 5:
                self.current_strategy = "epsilon_greedy"
                self.epsilon_greedy.epsilon = 0.05
            else:
                self.current_strategy = "boltzmann"

    def decay(self):
        """Decay exploration parameters"""
        self.epsilon_greedy.decay()
        self.boltzmann.decay()

    def reset(self):
        """Reset all strategies"""
        self.epsilon_greedy.reset()
        self.boltzmann.reset()
        self.ucb.reset()
        self.episode_count = 0
        self.performance_history = []


class ContextAwarePolicy(BasePolicy):
    """
    Context-aware policy that adapts exploration based on context
    Higher risk contexts use more exploitation (less exploration)
    """

    def __init__(self, base_epsilon: float = 0.2):
        """
        Args:
            base_epsilon: Base exploration rate
        """
        self.base_epsilon = base_epsilon

    def select_action(self, state, q_values: Dict, valid_actions: List[int],
                      context: Dict = None) -> int:
        """
        Select action with context-aware exploration

        Args:
            state: Current state
            q_values: Q-values for actions
            valid_actions: List of valid action indices
            context: Context dictionary with risk/conf scores
        """
        if not valid_actions:
            return 0

        # Adjust epsilon based on context
        epsilon = self.base_epsilon

        if context:
            risk_score = context.get("risk_score", 0.5)
            conf_score = context.get("conf_score", 0.5)

            # Higher risk/confidentiality = less exploration
            combined_score = (risk_score + conf_score) / 2.0
            epsilon = self.base_epsilon * (1.0 - combined_score)

        # Epsilon-greedy with adjusted epsilon
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # Exploitation
        if not q_values:
            return random.choice(valid_actions)

        valid_q = {a: q_values.get(a, 0.0) for a in valid_actions}
        return max(valid_q, key=valid_q.get)


class SafeExplorationPolicy(BasePolicy):
    """
    Safe exploration policy that avoids risky actions
    during exploration phase
    """

    def __init__(self, epsilon: float = 0.2, safety_threshold: float = 0.7):
        """
        Args:
            epsilon: Exploration rate
            safety_threshold: Minimum safety score for exploration
        """
        self.epsilon = epsilon
        self.safety_threshold = safety_threshold
        self.action_safety_scores = {}

    def update_safety_score(self, action: int, success: bool):
        """Update safety score for action based on outcome"""
        if action not in self.action_safety_scores:
            self.action_safety_scores[action] = 0.5

        # Exponential moving average
        alpha = 0.1
        new_score = 1.0 if success else 0.0
        self.action_safety_scores[action] = (
                (1 - alpha) * self.action_safety_scores[action] + alpha * new_score
        )

    def select_action(self, state, q_values: Dict, valid_actions: List[int]) -> int:
        """Select action with safety constraints"""
        if not valid_actions:
            return 0

        # Filter safe actions
        safe_actions = [
            a for a in valid_actions
            if self.action_safety_scores.get(a, 0.5) >= self.safety_threshold
        ]

        # If no safe actions, use all valid actions
        if not safe_actions:
            safe_actions = valid_actions

        # Epsilon-greedy over safe actions
        if random.random() < self.epsilon:
            return random.choice(safe_actions)

        if not q_values:
            return random.choice(safe_actions)

        safe_q = {a: q_values.get(a, 0.0) for a in safe_actions}
        return max(safe_q, key=safe_q.get)