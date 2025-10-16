"""
Enhanced RL Agent with Deep Q-Network (DQN) and Actor-Critic support
Based on research papers on adaptive security systems
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import random
from environment import CryptoAlgorithm, EnhancedEnvironment


class QTableAgent:
    def __init__(self, action_space_size: int, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.2):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

        self.episode_rewards = []
        self.episode_steps = 0

    def get_q_values(self, state: str) -> Dict[int, float]:
        if state not in self.q_table:
            self.q_table[state] = {i: 0.0 for i in range(self.action_space_size)}
        return self.q_table[state]

    def select_action(self, state: str, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.get_q_values(state)
        valid_q = {a: q_values[a] for a in valid_actions}

        if not valid_q:
            return random.choice(valid_actions)

        return max(valid_q, key=valid_q.get)

    def update(self, state: str, action: int, reward: float,
               next_state: str, done: bool = False):
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        old_value = q_values[action]

        if done:
            td_target = reward
        else:
            next_max = max(next_q_values.values()) if next_q_values else 0.0
            td_target = reward + self.gamma * next_max

        td_error = td_target - old_value

        new_value = old_value + self.alpha * td_error
        q_values[action] = new_value

        self.episode_rewards.append(reward)
        self.episode_steps += 1

    def end_episode(self):
        self.episode_steps = 0
        self.episode_rewards = []

        self.epsilon = max(0.01, self.epsilon * 0.995)

    def get_policy(self, state: str) -> Dict[int, float]:
        q_values = self.get_q_values(state)

        q_array = np.array(list(q_values.values()))
        exp_q = np.exp(q_array - np.max(q_array))  # Numerical stability
        probs = exp_q / np.sum(exp_q)

        return {i: probs[i] for i in range(len(probs))}


class DQNAgent:

    def __init__(self, state_dim: int, action_space_size: int,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 0.2,
                 buffer_size: int = 10000,
                 batch_size: int = 32):
        """
        Args:
            state_dim: Dimension of state vector
            action_space_size: Number of possible actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Exploration rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
        """
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.network_weights = self._initialize_network(state_dim, action_space_size, hidden_dims)
        self.target_network_weights = self._initialize_network(state_dim, action_space_size, hidden_dims)

        self.replay_buffer = deque(maxlen=buffer_size)

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
        output = np.random.randn(self.action_space_size) * 0.1
        return output

    def select_action(self, state: np.ndarray, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self._forward(state, self.network_weights)

        masked_q = np.full(self.action_space_size, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        # In production: compute loss and backpropagate
        # loss = MSE(Q(s,a), r + γ max_a' Q_target(s',a'))

        self.training_steps += 1

        if self.training_steps % 100 == 0:
            self._update_target_network()

        self.epsilon = max(0.01, self.epsilon * 0.995)

    def _update_target_network(self):
        self.target_network_weights = self.network_weights.copy()

    def end_episode(self):
        self.episode_rewards = []


class ActorCriticAgent:

    def __init__(self, state_dim: int, action_space_size: int,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.95):
        """
        Args:
            state_dim: Dimension of state vector
            action_space_size: Number of possible actions
            actor_lr: Learning rate for actor (policy)
            critic_lr: Learning rate for critic (value function)
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.gamma = gamma

        self.actor_weights = self._initialize_network(state_dim, action_space_size)

        self.critic_weights = self._initialize_network(state_dim, 1)

        self.episode_rewards = []

    def _initialize_network(self, input_dim: int, output_dim: int) -> Dict:
        return {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'initialized': True
        }

    def _actor_forward(self, state: np.ndarray) -> np.ndarray:
        logits = np.random.randn(self.action_space_size)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    def _critic_forward(self, state: np.ndarray) -> float:
        value = np.random.randn() * 0.1
        return value

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        if not valid_actions:
            return 0

        probs = self._actor_forward(state)

        masked_probs = np.zeros(self.action_space_size)
        for a in valid_actions:
            masked_probs[a] = probs[a]

        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            for a in valid_actions:
                masked_probs[a] = 1.0 / len(valid_actions)

        action = np.random.choice(self.action_space_size, p=masked_probs)
        return int(action)

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        value = self._critic_forward(state)
        next_value = 0.0 if done else self._critic_forward(next_state)
        td_target = reward + self.gamma * next_value
        advantage = td_target - value

        # In production:
        # 1. Update critic to minimize (td_target - value)^2
        # 2. Update actor using policy gradient with advantage

        self.episode_rewards.append(reward)

    def end_episode(self):
        self.episode_rewards = []


class HybridRLAgent:

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