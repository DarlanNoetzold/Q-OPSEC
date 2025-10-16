from typing import Dict, Any, List
from pathlib import Path
import json
from environment import (
    EnhancedEnvironment, CryptoAlgorithm, SecurityLevel,
    map_security_level
)
from agent import HybridRLAgent, QTableAgent
from policies import (
    EpsilonGreedyPolicy, BoltzmannPolicy, UCBPolicy,
    AdaptivePolicy, ContextAwarePolicy, SafeExplorationPolicy
)
from registry import RLRegistry


class ImprovedRLEngineService:
    """
    Enhanced RL Engine Service with improved modeling
    Maintains backward compatibility with existing API
    """

    def __init__(self, registry_path: Path = Path("./rl_registry.json"),
                 use_dqn: bool = False,
                 policy_type: str = "context_aware"):
        """
        Args:
            registry_path: Path to save/load Q-table
            use_dqn: Whether to use DQN (True) or Q-learning (False)
            policy_type: Type of policy to use
                - "epsilon_greedy": Standard epsilon-greedy
                - "boltzmann": Softmax exploration
                - "ucb": Upper Confidence Bound
                - "adaptive": Adaptive strategy switching
                - "context_aware": Context-aware exploration
                - "safe": Safe exploration with constraints
        """
        # Initialize environment
        self.env = EnhancedEnvironment()

        # Initialize agent
        state_dim = 18  # From compute_state_vector
        action_space_size = len(CryptoAlgorithm)
        self.agent = HybridRLAgent(state_dim, action_space_size, use_dqn=use_dqn)

        # Initialize policy
        self.policy = self._create_policy(policy_type)
        self.policy_type = policy_type

        # Registry for persistence
        self.registry = RLRegistry(registry_path)

        # Load saved Q-table if using Q-learning
        if not use_dqn:
            saved_q_table = self.registry.load()
            if saved_q_table and hasattr(self.agent.agent, 'q_table'):
                self.agent.agent.q_table = saved_q_table

        # Training mode flag
        self.training_mode = True

        # Episode tracking
        self.current_episode_experiences = []
        self.episode_count = 0

        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_negotiations': 0,
            'failed_negotiations': 0,
            'average_reward': 0.0,
            'algorithm_usage': {algo.value: 0 for algo in CryptoAlgorithm}
        }

    def _create_policy(self, policy_type: str):
        """Create policy based on type"""
        if policy_type == "epsilon_greedy":
            return EpsilonGreedyPolicy(epsilon=0.2)
        elif policy_type == "boltzmann":
            return BoltzmannPolicy(temperature=1.0)
        elif policy_type == "ucb":
            return UCBPolicy(c=2.0)
        elif policy_type == "adaptive":
            return AdaptivePolicy()
        elif policy_type == "context_aware":
            return ContextAwarePolicy(base_epsilon=0.2)
        elif policy_type == "safe":
            return SafeExplorationPolicy(epsilon=0.2)
        else:
            return EpsilonGreedyPolicy(epsilon=0.2)

    def decide_algorithms(self, context: Dict[str, Any]) -> List[str]:
        """
        Decide cryptographic algorithms based on context
        Uses RL agent to select optimal algorithm

        Args:
            context: Request context with risk/conf scores

        Returns:
            List of algorithm names
        """
        # Extract features from context
        features = self.env.extract_features(context)

        # Map to security level
        risk_score = context.get("risk_score", 0.5)
        conf_score = context.get("conf_score", 0.5)
        security_level = map_security_level(risk_score, conf_score)

        # Get valid actions
        valid_actions_enum = self.env.get_valid_actions(features, security_level)
        valid_action_indices = [self.env.actions.index(a) for a in valid_actions_enum]

        # Select action using agent and policy
        if self.policy_type == "context_aware":
            # Context-aware policy needs context
            state_hash = self.env.compute_state_hash(features)
            q_values = self.agent.get_q_table().get(state_hash, {})
            action_idx = self.policy.select_action(
                state_hash, q_values, valid_action_indices, context=context
            )
        else:
            # Other policies
            state_hash = self.env.compute_state_hash(features)
            q_values = self.agent.get_q_table().get(state_hash, {})
            action_idx = self.policy.select_action(
                state_hash, q_values, valid_action_indices
            )

        # Get selected algorithm
        selected_algo = self.env.actions[action_idx]

        # Build algorithm list based on selection
        algorithms = self._build_algorithm_list(selected_algo, features)

        # Store experience for later update
        if self.training_mode:
            self.current_episode_experiences.append({
                'state': state_hash,
                'action': action_idx,
                'features': features,
                'security_level': security_level,
                'context': context
            })

        # Update metrics
        self.metrics['total_requests'] += 1
        self.metrics['algorithm_usage'][selected_algo.value] += 1

        return algorithms

    def _build_algorithm_list(self, primary_algo: CryptoAlgorithm,
                              features) -> List[str]:
        """
        Build list of algorithms based on primary selection
        Includes fallbacks and compatibility layers
        """
        algorithms = []

        # Add primary algorithm
        algorithms.append(primary_algo.value)

        # Add fallback based on quantum availability
        if features.qkd_available:
            # Quantum available: add PQC as fallback
            if primary_algo not in [CryptoAlgorithm.PQC_KYBER,
                                    CryptoAlgorithm.PQC_DILITHIUM,
                                    CryptoAlgorithm.PQC_NTRU]:
                algorithms.append(CryptoAlgorithm.PQC_KYBER.value)

        # Always add classical fallback
        if CryptoAlgorithm.AES_256_GCM.value not in algorithms:
            algorithms.append(CryptoAlgorithm.AES_256_GCM.value)

        return algorithms

    def build_negotiation_payload(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build negotiation payload for handshake
        Maintains API compatibility

        Args:
            req: Request dictionary with context

        Returns:
            Negotiation payload
        """
        # Decide algorithms using RL
        proposed_algorithms = self.decide_algorithms(req)

        # Build payload
        payload = {
            "request_id": req["request_id"],
            "source": req["source"],
            "destination": req["destination"],
            "dst_props": req.get("dst_props", {}),
            "proposed": proposed_algorithms,
            "security_level": self._get_security_level_name(req),
            "metadata": {
                "rl_engine_version": "2.0",
                "policy_type": self.policy_type,
                "training_mode": self.training_mode
            }
        }

        return payload

    def _get_security_level_name(self, context: Dict[str, Any]) -> str:
        """Get security level name from context"""
        risk_score = context.get("risk_score", 0.5)
        conf_score = context.get("conf_score", 0.5)
        security_level = map_security_level(risk_score, conf_score)
        return security_level.name

    def process_feedback(self, request_id: str, outcome: Dict[str, Any]):
        """
        Process feedback from negotiation outcome
        Updates RL agent based on reward

        Args:
            request_id: Request identifier
            outcome: Outcome dictionary with success, latency, etc.
        """
        if not self.training_mode or not self.current_episode_experiences:
            return

        # Get last experience (assumes feedback is for last request)
        experience = self.current_episode_experiences[-1]

        # Compute reward
        reward = self.env.compute_reward(
            self.env.actions[experience['action']],
            experience['features'],
            experience['security_level'],
            outcome
        )

        # Update agent
        state = experience['state']
        action = experience['action']

        # For Q-learning, we need next state (use current state as approximation)
        next_state = state
        done = False

        self.agent.update(state, action, reward, next_state, done)

        # Update environment performance metrics
        self.env.update_algorithm_performance(
            self.env.actions[action],
            outcome
        )

        # Update policy if safe exploration
        if isinstance(self.policy, SafeExplorationPolicy):
            success = outcome.get("success", False)
            self.policy.update_safety_score(action, success)

        # Update metrics
        if outcome.get("success", False):
            self.metrics['successful_negotiations'] += 1
        else:
            self.metrics['failed_negotiations'] += 1

        # Update average reward
        alpha = 0.1
        self.metrics['average_reward'] = (
                (1 - alpha) * self.metrics['average_reward'] + alpha * reward
        )

    def end_episode(self):
        """
        Mark end of training episode
        Saves Q-table and decays exploration
        """
        if not self.training_mode:
            return

        # End episode for agent
        self.agent.end_episode()

        # Decay policy exploration
        if hasattr(self.policy, 'decay'):
            self.policy.decay()

        # Update adaptive policy
        if isinstance(self.policy, AdaptivePolicy):
            episode_reward = sum(
                exp.get('reward', 0) for exp in self.current_episode_experiences
            )
            self.policy.update_strategy(episode_reward)

        # Save Q-table
        q_table = self.agent.get_q_table()
        if q_table:
            self.registry.save(q_table)

        # Clear episode experiences
        self.current_episode_experiences = []
        self.episode_count += 1

    def set_training_mode(self, enabled: bool):
        """Enable or disable training mode"""
        self.training_mode = enabled

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    def get_q_table_stats(self) -> Dict[str, Any]:
        """Get Q-table statistics"""
        q_table = self.agent.get_q_table()

        if not q_table:
            return {
                'num_states': 0,
                'num_state_action_pairs': 0,
                'avg_q_value': 0.0
            }

        num_states = len(q_table)
        num_pairs = sum(len(actions) for actions in q_table.values())

        all_q_values = [
            q for actions in q_table.values() for q in actions.values()
        ]
        avg_q = sum(all_q_values) / len(all_q_values) if all_q_values else 0.0

        return {
            'num_states': num_states,
            'num_state_action_pairs': num_pairs,
            'avg_q_value': avg_q,
            'max_q_value': max(all_q_values) if all_q_values else 0.0,
            'min_q_value': min(all_q_values) if all_q_values else 0.0
        }

    def export_policy(self, path: Path):
        """Export learned policy to file"""
        q_table = self.agent.get_q_table()

        policy_data = {
            'policy_type': self.policy_type,
            'q_table': q_table,
            'metrics': self.metrics,
            'episode_count': self.episode_count,
            'algorithm_performance': {
                algo.value: perf
                for algo, perf in self.env.algorithm_performance.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(policy_data, f, indent=2)

    def import_policy(self, path: Path):
        """Import learned policy from file"""
        with open(path, 'r') as f:
            policy_data = json.load(f)

        # Load Q-table
        if 'q_table' in policy_data and hasattr(self.agent.agent, 'q_table'):
            self.agent.agent.q_table = policy_data['q_table']

        # Load metrics
        if 'metrics' in policy_data:
            self.metrics.update(policy_data['metrics'])

        # Load episode count
        if 'episode_count' in policy_data:
            self.episode_count = policy_data['episode_count']