from typing import Dict, Any, List
from pathlib import Path
from environment import extract_state, ACTIONS
from policies import EpsilonGreedyPolicy
from agent import RLAgent
from registry import RLRegistry

class RLEngineService:
    def __init__(self, registry_path: Path = Path("./rl_registry.json")):
        self.policy = EpsilonGreedyPolicy(ACTIONS, epsilon=0.2)
        self.agent = RLAgent()
        self.registry = RLRegistry(registry_path)
        self.q_table = self.registry.load() or {}

    def decide_algorithms(self, context: Dict[str, Any]) -> List[str]:
        state = extract_state(context)
        action_id = self.policy.select_action(state, self.q_table)
        primary = ACTIONS[action_id]
        props = context.get("dst_props", {})
        hw = props.get("hardware", []) or []
        algos: List[str] = []
        if "QKD" in [h.upper() for h in hw]:
            if primary in ["E91", "BB84"]:
                algos.append(f"QKD_{primary}")
            else:
                algos.append("QKD_E91")
            algos.append("AES256_GCM")
        else:
            algos.append("AES256_GCM")
            algos.append("RSA_PQ")
        return algos

    def build_negotiation_payload(self, req: Dict[str, Any]) -> Dict[str, Any]:
        proposed = self.decide_algorithms(req)
        return {
            "request_id": req["request_id"],
            "source": req["source"],
            "destination": req["destination"],
            "dst_props": req.get("dst_props", {}),
            "proposed": proposed
        }