"""
Enhanced RL Agent with FORCED EXPLORATION and Balanced Selection
Fixed to prevent algorithm monopoly
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import random
from environment import CryptoAlgorithm, EnhancedEnvironment


class QTableAgent:
    def __init__(self, action_space_size: int, alpha: float = 0.15,
                 gamma: float = 0.95, epsilon: float = 0.5,  # INCREASED epsilon
                 epsilon_min: float = 0.15,  # HIGHER minimum
                 epsilon_decay: float = 0.998):  # SLOWER decay
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space_size = action_space_size

        # Track algorithm usage to force diversity
        self.action_counts = {i: 0 for i in range(action_space_size)}
        self.total_actions = 0

        self.episode_rewards = []
        self.episode_steps = 0

    def get_q_values(self, state: str) -> Dict[int, float]:
        if state not in self.q_table:
            # Initialize with small random values for exploration
            self.q_table[state] = {i: np.random.uniform(-0.1, 0.1)
                                   for i in range(self.action_space_size)}
        return self.q_table[state]

    def select_action(self, state: str, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        # FORCED DIVERSITY: Penalize overused algorithms
        usage_penalty = self._compute_usage_penalty()

        # Epsilon-greedy with diversity bonus
        if explore and random.random() < self.epsilon:
            # Weighted random selection favoring less-used algorithms
            weights = [1.0 / (1.0 + usage_penalty[a]) for a in valid_actions]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            return np.random.choice(valid_actions, p=weights)

        # Exploitation with diversity consideration
        q_values = self.get_q_values(state)

        # Adjust Q-values with diversity bonus
        adjusted_q = {}
        for a in valid_actions:
            diversity_bonus = -usage_penalty[a] * 0.5  # Penalty for overuse
            adjusted_q[a] = q_values[a] + diversity_bonus

        if not adjusted_q:
            return random.choice(valid_actions)

        # Add noise to break ties
        best_actions = []
        max_q = max(adjusted_q.values())
        for a, q in adjusted_q.items():
            if abs(q - max_q) < 0.01:  # Consider similar Q-values
                best_actions.append(a)

        return random.choice(best_actions) if best_actions else max(adjusted_q, key=adjusted_q.get)

    def _compute_usage_penalty(self) -> Dict[int, float]:
        """Compute penalty for overused algorithms"""
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_usage = self.total_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.action_counts[i]
            # Penalty grows quadratically with overuse
            if actual_usage > expected_usage:
                penalties[i] = ((actual_usage - expected_usage) / expected_usage) ** 1.5
            else:
                penalties[i] = 0.0

        return penalties

    def update(self, state: str, action: int, reward: float,
               next_state: str, done: bool = False):
        # Track usage
        self.action_counts[action] += 1
        self.total_actions += 1

        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        old_value = q_values[action]

        if done:
            td_target = reward
        else:
            next_max = max(next_q_values.values()) if next_q_values else 0.0
            td_target = reward + self.gamma * next_max

        td_error = td_target - old_value

        # Update with learning rate
        new_value = old_value + self.alpha * td_error
        q_values[action] = new_value

        self.episode_rewards.append(reward)
        self.episode_steps += 1

    def end_episode(self):
        self.episode_steps = 0
        self.episode_rewards = []

        # Decay epsilon more slowly
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state: str) -> Dict[int, float]:
        q_values = self.get_q_values(state)

        q_array = np.array(list(q_values.values()))
        exp_q = np.exp(q_array - np.max(q_array))
        probs = exp_q / np.sum(exp_q)

        return {i: probs[i] for i in range(len(probs))}

    def get_usage_stats(self) -> Dict[int, float]:
        """Get algorithm usage statistics"""
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        return {i: count / self.total_actions
                for i, count in self.action_counts.items()}


class DQNAgent:
    """DQN with forced exploration and diversity"""

    def __init__(self, state_dim: int, action_space_size: int,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.5,  # INCREASED
                 epsilon_min: float = 0.15,  # HIGHER minimum
                 epsilon_decay: float = 0.998,  # SLOWER decay
                 buffer_size: int = 10000,
                 batch_size: int = 32):

        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.network_weights = self._initialize_network(state_dim, action_space_size, hidden_dims)
        self.target_network_weights = self._initialize_network(state_dim, action_space_size, hidden_dims)

        self.replay_buffer = deque(maxlen=buffer_size)

        # Track usage for diversity
        self.action_counts = {i: 0 for i in range(action_space_size)}
        self.total_actions = 0

        self.training_steps = 0
        self.episode_rewards = []

    def _initialize_network(self, input_dim: int, output_dim: int,
                            hidden_dims: List[int]) -> Dict:
        weights = {
            'layer_dims': [input_dim] + hidden_dims + [output_dim],
            'initialized': True
        }
        return weights

    def _forward(self, state: np.ndarray, weights: Dict) -> np.ndarray:
        # Simplified forward pass with random initialization
        output = np.random.randn(self.action_space_size) * 0.5
        return output

    def select_action(self, state: np.ndarray, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        # FORCED DIVERSITY
        usage_penalty = self._compute_usage_penalty()

        if explore and random.random() < self.epsilon:
            # Weighted random favoring less-used algorithms
            weights = [1.0 / (1.0 + usage_penalty[a]) for a in valid_actions]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            return np.random.choice(valid_actions, p=weights)

        q_values = self._forward(state, self.network_weights)

        # Apply diversity bonus
        masked_q = np.full(self.action_space_size, -np.inf)
        for a in valid_actions:
            diversity_bonus = -usage_penalty[a] * 0.5
            masked_q[a] = q_values[a] + diversity_bonus

        return int(np.argmax(masked_q))

    def _compute_usage_penalty(self) -> Dict[int, float]:
        """Compute penalty for overused algorithms"""
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_usage = self.total_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.action_counts[i]
            if actual_usage > expected_usage:
                penalties[i] = ((actual_usage - expected_usage) / expected_usage) ** 1.5
            else:
                penalties[i] = 0.0

        return penalties

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

        # Track usage
        self.action_counts[action] += 1
        self.total_actions += 1

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        # In production: compute loss and backpropagate
        self.training_steps += 1

        if self.training_steps % 100 == 0:
            self._update_target_network()

        # Decay epsilon more slowly
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_target_network(self):
        self.target_network_weights = self.network_weights.copy()

    def end_episode(self):
        self.episode_rewards = []


class ActorCriticAgent:
    """Actor-Critic with diversity enforcement"""

    def __init__(self, state_dim: int, action_space_size: int,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.95,
                 entropy_coef: float = 0.05):  # Entropy bonus for exploration

        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.actor_weights = self._initialize_network(state_dim, action_space_size)
        self.critic_weights = self._initialize_network(state_dim, 1)

        # Track usage
        self.action_counts = {i: 0 for i in range(action_space_size)}
        self.total_actions = 0

        self.episode_rewards = []

    def _initialize_network(self, input_dim: int, output_dim: int) -> Dict:
        return {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'initialized': True
        }

    def _actor_forward(self, state: np.ndarray) -> np.ndarray:
        # Random logits with diversity consideration
        logits = np.random.randn(self.action_space_size) * 0.5

        # Add diversity bonus
        usage_penalty = self._compute_usage_penalty()
        for i in range(self.action_space_size):
            logits[i] -= usage_penalty[i] * 0.3

        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    def _critic_forward(self, state: np.ndarray) -> float:
        value = np.random.randn() * 0.1
        return value

    def _compute_usage_penalty(self) -> Dict[int, float]:
        """Compute penalty for overused algorithms"""
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_usage = self.total_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.action_counts[i]
            if actual_usage > expected_usage:
                penalties[i] = ((actual_usage - expected_usage) / expected_usage) ** 1.5
            else:
                penalties[i] = 0.0

        return penalties

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        probs = self._actor_forward(state)

        # Mask invalid actions
        masked_probs = np.zeros(self.action_space_size)
        for a in valid_actions:
            masked_probs[a] = probs[a]

        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            for a in valid_actions:
                masked_probs[a] = 1.0 / len(valid_actions)

        action = np.random.choice(self.action_space_size, p=masked_probs)

        # Track usage
        self.action_counts[action] += 1
        self.total_actions += 1

        return int(action)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        value = self._critic_forward(state)
        next_value = 0.0 if done else self._critic_forward(next_state)
        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        self.episode_rewards.append(reward)

    def end_episode(self):
        self.episode_rewards = []


class HybridRLAgent:
    """Hybrid agent with forced diversity"""

    def __init__(self, state_dim: int, action_space_size: int,
                 use_dqn: bool = False):
        self.use_dqn = use_dqn
        self.action_space_size = action_space_size

        if use_dqn:
            self.agent = DQNAgent(state_dim, action_space_size)
        else:
            self.agent = QTableAgent(action_space_size)

    def select_action(self, state, valid_actions: List[int],
                      explore: bool = True) -> int:
        return self.agent.select_action(state, valid_actions, explore)

    def update(self, state, action: int, reward: float, next_state, done: bool = False):
        if self.use_dqn:
            self.agent.store_experience(state, action, reward, next_state, done)
            self.agent.train_step()
        else:
            self.agent.update(state, action, reward, next_state, done)

    def end_episode(self):
        self.agent.end_episode()

    def get_q_table(self) -> Dict:
        if not self.use_dqn and hasattr(self.agent, 'q_table'):
            return self.agent.q_table
        return {}

    def get_usage_stats(self) -> Dict[int, float]:
        """Get algorithm usage statistics"""
        if hasattr(self.agent, 'get_usage_stats'):
            return self.agent.get_usage_stats()
        elif hasattr(self.agent, 'action_counts'):
            total = self.agent.total_actions
            if total == 0:
                return {i: 0.0 for i in range(self.action_space_size)}
            return {i: count / total for i, count in self.agent.action_counts.items()}
        return {}