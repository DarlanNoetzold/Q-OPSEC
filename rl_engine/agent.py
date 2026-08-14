
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque
import random
from environment import CryptoAlgorithm, EnhancedEnvironment


class QTableAgent:
    def __init__(self, action_space_size: int, alpha: float = 0.12,
                 gamma: float = 0.93, epsilon: float = 0.70,
                 epsilon_min: float = 0.25,
                 epsilon_decay: float = 0.9995):
        self.q_table: Dict[str, Dict[int, float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.action_space_size = action_space_size

        self.action_counts = {i: 0 for i in range(action_space_size)}
        self.recent_actions = deque(maxlen=100)
        self.episode_action_counts = {i: 0 for i in range(action_space_size)}
        self.total_actions = 0
        self.episode_actions = 0

        self.steps_since_forced_exploration = 0
        self.forced_exploration_interval = 25

        self.episode_rewards = []
        self.episode_steps = 0

    def get_q_values(self, state: str) -> Dict[int, float]:
        if state not in self.q_table:
            self.q_table[state] = {i: np.random.uniform(-0.05, 0.05)
                                   for i in range(self.action_space_size)}
        return self.q_table[state]

    def select_action(self, state: str, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        self.steps_since_forced_exploration += 1

        if self.steps_since_forced_exploration >= self.forced_exploration_interval:
            self.steps_since_forced_exploration = 0
            usage_counts = {a: self.action_counts[a] for a in valid_actions}
            min_usage = min(usage_counts.values())
            least_used = [a for a, c in usage_counts.items() if c == min_usage]
            return random.choice(least_used)

        usage_penalty = self._compute_multi_level_penalty()
        recent_penalty = self._compute_recent_usage_penalty()
        episode_penalty = self._compute_episode_penalty()

        total_penalty = {}
        for a in valid_actions:
            total_penalty[a] = (usage_penalty[a] * 0.4 +
                                recent_penalty[a] * 0.3 +
                                episode_penalty[a] * 0.3)

        adaptive_epsilon = self._compute_adaptive_epsilon()
        effective_epsilon = max(self.epsilon, adaptive_epsilon)

        if explore and random.random() < effective_epsilon:
            weights = []
            for a in valid_actions:
                penalty_factor = np.exp(-total_penalty[a] * 2.0)
                weights.append(penalty_factor)

            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return np.random.choice(valid_actions, p=weights)
            else:
                return random.choice(valid_actions)

        q_values = self.get_q_values(state)

        adjusted_q = {}
        for a in valid_actions:
            diversity_adjustment = -total_penalty[a] * 1.5

            if self.action_counts[a] == 0:
                diversity_adjustment += 2.0
            elif self.action_counts[a] < self.total_actions / (self.action_space_size * 2):
                diversity_adjustment += 1.0

            adjusted_q[a] = q_values[a] + diversity_adjustment

        if not adjusted_q:
            return random.choice(valid_actions)

        q_array = np.array([adjusted_q[a] for a in valid_actions])

        temperature = 0.5
        exp_q = np.exp(q_array / temperature)
        probs = exp_q / np.sum(exp_q)

        return np.random.choice(valid_actions, p=probs)

    def _compute_multi_level_penalty(self) -> Dict[int, float]:
        
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_usage = self.total_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.action_counts[i]

            if actual_usage > expected_usage * 2.5:
                penalties[i] = 5.0
            elif actual_usage > expected_usage * 2.0:
                penalties[i] = 3.0
            elif actual_usage > expected_usage * 1.5:
                penalties[i] = 1.5
            elif actual_usage > expected_usage * 1.2:
                penalties[i] = 0.5
            else:
                penalties[i] = 0.0

        return penalties

    def _compute_recent_usage_penalty(self) -> Dict[int, float]:
        
        if len(self.recent_actions) == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        recent_counts = {}

        for idx, action in enumerate(self.recent_actions):
            weight = (idx + 1) / len(self.recent_actions)
            recent_counts[action] = recent_counts.get(action, 0) + weight

        max_recent = max(recent_counts.values()) if recent_counts else 1.0

        for i in range(self.action_space_size):
            recent_usage = recent_counts.get(i, 0)
            penalties[i] = (recent_usage / max_recent) * 2.0

        return penalties

    def _compute_episode_penalty(self) -> Dict[int, float]:
        
        if self.episode_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_episode_usage = self.episode_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.episode_action_counts[i]

            if actual_usage > expected_episode_usage * 2.0:
                penalties[i] = 2.0
            elif actual_usage > expected_episode_usage * 1.5:
                penalties[i] = 1.0
            else:
                penalties[i] = 0.0

        return penalties

    def _compute_adaptive_epsilon(self) -> float:
        
        if self.total_actions < 50:
            return self.epsilon

        counts = list(self.action_counts.values())
        mean_usage = np.mean(counts)

        if mean_usage == 0:
            return self.epsilon

        variance = np.var(counts)
        cv = np.sqrt(variance) / mean_usage

        if cv > 0.8:
            return min(0.9, self.epsilon + 0.3)
        elif cv > 0.5:
            return min(0.8, self.epsilon + 0.2)
        elif cv > 0.3:
            return min(0.7, self.epsilon + 0.1)
        else:
            return self.epsilon

    def update(self, state: str, action: int, reward: float,
               next_state: str, done: bool = False):
        self.action_counts[action] += 1
        self.episode_action_counts[action] += 1
        self.recent_actions.append(action)
        self.total_actions += 1
        self.episode_actions += 1

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
        self.episode_action_counts = {i: 0 for i in range(self.action_space_size)}
        self.episode_actions = 0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self, state: str) -> Dict[int, float]:
        q_values = self.get_q_values(state)

        q_array = np.array(list(q_values.values()))
        exp_q = np.exp(q_array - np.max(q_array))
        probs = exp_q / np.sum(exp_q)

        return {i: probs[i] for i in range(len(probs))}

    def get_usage_stats(self) -> Dict[int, float]:
        
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        return {i: count / self.total_actions
                for i, count in self.action_counts.items()}

    def get_diversity_metrics(self) -> Dict[str, float]:
        
        if self.total_actions == 0:
            return {
                'variance': 0.0,
                'cv': 0.0,
                'gini': 0.0,
                'entropy': 0.0
            }

        counts = np.array(list(self.action_counts.values()))

        variance = np.var(counts)

        mean_usage = np.mean(counts)
        cv = np.sqrt(variance) / mean_usage if mean_usage > 0 else 0

        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        return {
            'variance': float(variance),
            'cv': float(cv),
            'gini': float(gini),
            'entropy': float(entropy)
        }


class DQNAgent:
    

    def __init__(self, state_dim: int, action_space_size: int,
                 hidden_dims: List[int] = [128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.93,
                 epsilon: float = 0.70,
                 epsilon_min: float = 0.25,
                 epsilon_decay: float = 0.9995,
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

        self.action_counts = {i: 0 for i in range(action_space_size)}
        self.recent_actions = deque(maxlen=100)
        self.episode_action_counts = {i: 0 for i in range(action_space_size)}
        self.total_actions = 0
        self.episode_actions = 0

        self.steps_since_forced_exploration = 0
        self.forced_exploration_interval = 25

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
        output = np.random.randn(self.action_space_size) * 0.3
        return output

    def select_action(self, state: np.ndarray, valid_actions: List[int],
                      explore: bool = True) -> int:
        if not valid_actions:
            return 0

        self.steps_since_forced_exploration += 1

        if self.steps_since_forced_exploration >= self.forced_exploration_interval:
            self.steps_since_forced_exploration = 0
            usage_counts = {a: self.action_counts[a] for a in valid_actions}
            min_usage = min(usage_counts.values())
            least_used = [a for a, c in usage_counts.items() if c == min_usage]
            return random.choice(least_used)

        usage_penalty = self._compute_multi_level_penalty()
        recent_penalty = self._compute_recent_usage_penalty()
        episode_penalty = self._compute_episode_penalty()

        total_penalty = {}
        for a in valid_actions:
            total_penalty[a] = (usage_penalty[a] * 0.4 +
                                recent_penalty[a] * 0.3 +
                                episode_penalty[a] * 0.3)

        adaptive_epsilon = self._compute_adaptive_epsilon()
        effective_epsilon = max(self.epsilon, adaptive_epsilon)

        if explore and random.random() < effective_epsilon:
            weights = []
            for a in valid_actions:
                penalty_factor = np.exp(-total_penalty[a] * 2.0)
                weights.append(penalty_factor)

            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                return np.random.choice(valid_actions, p=weights)
            else:
                return random.choice(valid_actions)

        q_values = self._forward(state, self.network_weights)

        adjusted_q = {}
        for a in valid_actions:
            diversity_adjustment = -total_penalty[a] * 1.5

            if self.action_counts[a] == 0:
                diversity_adjustment += 2.0
            elif self.action_counts[a] < self.total_actions / (self.action_space_size * 2):
                diversity_adjustment += 1.0

            adjusted_q[a] = q_values[a] + diversity_adjustment

        q_array = np.array([adjusted_q[a] for a in valid_actions])
        temperature = 0.5
        exp_q = np.exp(q_array / temperature)
        probs = exp_q / np.sum(exp_q)

        return np.random.choice(valid_actions, p=probs)

    def _compute_multi_level_penalty(self) -> Dict[int, float]:
        if self.total_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_usage = self.total_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.action_counts[i]

            if actual_usage > expected_usage * 2.5:
                penalties[i] = 5.0
            elif actual_usage > expected_usage * 2.0:
                penalties[i] = 3.0
            elif actual_usage > expected_usage * 1.5:
                penalties[i] = 1.5
            elif actual_usage > expected_usage * 1.2:
                penalties[i] = 0.5
            else:
                penalties[i] = 0.0

        return penalties

    def _compute_recent_usage_penalty(self) -> Dict[int, float]:
        if len(self.recent_actions) == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        recent_counts = {}

        for idx, action in enumerate(self.recent_actions):
            weight = (idx + 1) / len(self.recent_actions)
            recent_counts[action] = recent_counts.get(action, 0) + weight

        max_recent = max(recent_counts.values()) if recent_counts else 1.0

        for i in range(self.action_space_size):
            recent_usage = recent_counts.get(i, 0)
            penalties[i] = (recent_usage / max_recent) * 2.0

        return penalties

    def _compute_episode_penalty(self) -> Dict[int, float]:
        if self.episode_actions == 0:
            return {i: 0.0 for i in range(self.action_space_size)}

        penalties = {}
        expected_episode_usage = self.episode_actions / self.action_space_size

        for i in range(self.action_space_size):
            actual_usage = self.episode_action_counts[i]

            if actual_usage > expected_episode_usage * 2.0:
                penalties[i] = 2.0
            elif actual_usage > expected_episode_usage * 1.5:
                penalties[i] = 1.0
            else:
                penalties[i] = 0.0

        return penalties

    def _compute_adaptive_epsilon(self) -> float:
        if self.total_actions < 50:
            return self.epsilon

        counts = list(self.action_counts.values())
        mean_usage = np.mean(counts)

        if mean_usage == 0:
            return self.epsilon

        variance = np.var(counts)
        cv = np.sqrt(variance) / mean_usage

        if cv > 0.8:
            return min(0.9, self.epsilon + 0.3)
        elif cv > 0.5:
            return min(0.8, self.epsilon + 0.2)
        elif cv > 0.3:
            return min(0.7, self.epsilon + 0.1)
        else:
            return self.epsilon

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

        self.action_counts[action] += 1
        self.episode_action_counts[action] += 1
        self.recent_actions.append(action)
        self.total_actions += 1
        self.episode_actions += 1

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        self.training_steps += 1

        if self.training_steps % 100 == 0:
            self._update_target_network()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_target_network(self):
        self.target_network_weights = self.network_weights.copy()

    def end_episode(self):
        self.episode_rewards = []
        self.episode_action_counts = {i: 0 for i in range(self.action_space_size)}
        self.episode_actions = 0


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

    def get_usage_stats(self) -> Dict[int, float]:
        
        if hasattr(self.agent, 'get_usage_stats'):
            return self.agent.get_usage_stats()
        elif hasattr(self.agent, 'action_counts'):
            total = self.agent.total_actions
            if total == 0:
                return {i: 0.0 for i in range(self.action_space_size)}
            return {i: count / total for i, count in self.agent.action_counts.items()}
        return {}

    def get_diversity_metrics(self) -> Dict[str, float]:
        
        if hasattr(self.agent, 'get_diversity_metrics'):
            return self.agent.get_diversity_metrics()
        return {}