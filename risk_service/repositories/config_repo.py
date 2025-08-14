class ConfigRepository:
    def get_policy_thresholds(self):
        return {
            "very_low": 0.15,
            "low": 0.35,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9
        }

    def get_policy_overrides(self, level: str):
        table = {
            "very_low": [],
            "low": [],
            "medium": ["enforce_mtls"],
            "high": ["enforce_mtls", "rotate_keys_24h", "pqc_required"],
            "critical": ["enforce_mtls", "rotate_keys_6h", "pqc_required", "block_high_risk_ops"]
        }
        return table.get(level, [])