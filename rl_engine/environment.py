from typing import Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass
from enum import Enum


class CryptoAlgorithm(Enum):
    QKD_BB84 = "QKD_BB84"
    QKD_E91 = "QKD_E91"
    QKD_CV = "QKD_CV-QKD"
    QKD_MDI = "QKD_MDI-QKD"
    QKD_DECOY = "QKD_DECOY"

    PQC_KYBER = "PQC_KYBER"
    PQC_DILITHIUM = "PQC_DILITHIUM"
    PQC_NTRU = "PQC_NTRU"
    PQC_SABER = "PQC_SABER"
    PQC_FALCON = "PQC_FALCON"

    HYBRID_RSA_PQC = "HYBRID_RSA_PQC"
    HYBRID_ECC_PQC = "HYBRID_ECC_PQC"

    AES_256_GCM = "AES_256_GCM"
    AES_192 = "AES_192"
    RSA_4096 = "RSA_4096"
    ECC_521 = "ECC_521"

    FALLBACK_AES = "FALLBACK_AES"


class SecurityLevel(Enum):
    VERY_LOW = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    VERY_HIGH = 4
    ULTRA = 5


@dataclass
class ContextFeatures:
    source_reputation: float
    source_location_risk: float
    source_device_type: str
    source_behavioral_score: float

    dest_reputation: float
    dest_location_risk: float
    dest_security_policy: str
    dest_hardware_capabilities: List[str]

    data_sensitivity: float
    data_type: str
    service_criticality: float

    time_of_day: int
    day_of_week: int
    is_peak_attack_time: bool

    current_threat_level: float
    incident_history_score: float
    anomaly_score: float

    system_load: float
    available_resources: float
    network_latency: float

    qkd_available: bool
    quantum_hardware_present: bool


class EnhancedEnvironment:
    def __init__(self):
        self.actions = list(CryptoAlgorithm)
        self.action_space_size = len(self.actions)
        self.security_levels = list(SecurityLevel)

        self.algorithm_requirements = self._build_algorithm_requirements()

        self.algorithm_performance = {
            algo: {
                'latency': 0.0,
                'success_rate': 1.0,
                'resource_cost': 0.5,
                'security_strength': 0.5
            } for algo in CryptoAlgorithm
        }

    def _build_algorithm_requirements(self) -> Dict[CryptoAlgorithm, Dict[str, Any]]:
        return {
            CryptoAlgorithm.QKD_BB84: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': True,
                'quantum_hardware': True,
                'min_resources': 0.7
            },
            CryptoAlgorithm.QKD_E91: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': True,
                'quantum_hardware': True,
                'min_resources': 0.7
            },
            CryptoAlgorithm.QKD_CV: {
                'min_security_level': SecurityLevel.VERY_HIGH,
                'requires_qkd': True,
                'quantum_hardware': True,
                'min_resources': 0.8
            },
            CryptoAlgorithm.QKD_MDI: {
                'min_security_level': SecurityLevel.VERY_HIGH,
                'requires_qkd': True,
                'quantum_hardware': True,
                'min_resources': 0.8
            },
            CryptoAlgorithm.QKD_DECOY: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': True,
                'quantum_hardware': True,
                'min_resources': 0.7
            },

            CryptoAlgorithm.PQC_KYBER: {
                'min_security_level': SecurityLevel.LOW,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.1
            },
            CryptoAlgorithm.PQC_DILITHIUM: {
                'min_security_level': SecurityLevel.LOW,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.1
            },
            CryptoAlgorithm.PQC_NTRU: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.4
            },
            CryptoAlgorithm.PQC_SABER: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.4
            },
            CryptoAlgorithm.PQC_FALCON: {
                'min_security_level': SecurityLevel.VERY_HIGH,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.5
            },

            CryptoAlgorithm.HYBRID_RSA_PQC: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.5
            },
            CryptoAlgorithm.HYBRID_ECC_PQC: {
                'min_security_level': SecurityLevel.HIGH,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.5
            },

            CryptoAlgorithm.AES_256_GCM: {
                'min_security_level': SecurityLevel.MODERATE,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.3
            },
            CryptoAlgorithm.AES_192: {
                'min_security_level': SecurityLevel.LOW,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.2
            },
            CryptoAlgorithm.RSA_4096: {
                'min_security_level': SecurityLevel.MODERATE,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.3
            },
            CryptoAlgorithm.ECC_521: {
                'min_security_level': SecurityLevel.MODERATE,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.3
            },

            CryptoAlgorithm.FALLBACK_AES: {
                'min_security_level': SecurityLevel.VERY_LOW,
                'requires_qkd': False,
                'quantum_hardware': False,
                'min_resources': 0.1
            }
        }

    def extract_features(self, context: Dict[str, Any]) -> ContextFeatures:
        
        dst_props = context.get("dst_props", {})
        hardware = dst_props.get("hardware", [])

        hw_list = [h.upper() for h in (hardware if isinstance(hardware, list) else [])]

        def get_or_default(key, default):
            value = context.get(key)
            return default if value is None else value

        return ContextFeatures(
            source_reputation=get_or_default("source_reputation", 0.5),
            source_location_risk=get_or_default("source_location_risk", 0.5),
            source_device_type=context.get("source", "unknown"),
            source_behavioral_score=get_or_default("source_behavioral_score", 0.5),

            dest_reputation=dst_props.get("reputation", 0.5),
            dest_location_risk=dst_props.get("location_risk", 0.5),
            dest_security_policy=dst_props.get("security_policy", "standard"),
            dest_hardware_capabilities=hw_list,

            data_sensitivity=get_or_default("conf_score", 0.5),
            data_type=context.get("data_type", "general"),
            service_criticality=get_or_default("service_criticality", 0.5),

            time_of_day=get_or_default("time_of_day", 12),
            day_of_week=get_or_default("day_of_day", 0),
            is_peak_attack_time=get_or_default("is_peak_attack_time", False),

            current_threat_level=get_or_default("risk_score", 0.5),
            incident_history_score=get_or_default("incident_history_score", 0.0),
            anomaly_score=get_or_default("anomaly_score", 0.0),

            system_load=get_or_default("system_load", 0.5),
            available_resources=get_or_default("available_resources", 1.0),
            network_latency=get_or_default("network_latency", 50.0),

            qkd_available="QKD" in hw_list,
            quantum_hardware_present="QUANTUM" in hw_list or "QKD" in hw_list
        )

    def compute_state_vector(self, features: ContextFeatures) -> np.ndarray:
        state_vector = np.array([
            features.source_reputation,
            features.source_location_risk,
            features.source_behavioral_score,
            features.dest_reputation,
            features.dest_location_risk,
            features.data_sensitivity,
            features.service_criticality,
            features.time_of_day / 24.0,
            features.day_of_week / 7.0,
            float(features.is_peak_attack_time),
            features.current_threat_level,
            features.incident_history_score,
            features.anomaly_score,
            features.system_load,
            features.available_resources,
            min(features.network_latency / 1000.0, 1.0),
            float(features.qkd_available),
            float(features.quantum_hardware_present)
        ], dtype=np.float32)

        return state_vector

    def compute_state_hash(self, features: ContextFeatures) -> str:
        risk_level = self._discretize(features.current_threat_level, bins=10)
        conf_level = self._discretize(features.data_sensitivity, bins=10)
        resource_level = self._discretize(features.available_resources, bins=5)

        quantum_status = "Q" if features.qkd_available else "C"
        criticality = "CRIT" if features.service_criticality > 0.7 else "NORM"
        time_risk = "PEAK" if features.is_peak_attack_time else "NORMAL"

        state_hash = f"R{risk_level}|C{conf_level}|Res{resource_level}|{quantum_status}|{criticality}|{time_risk}"

        return state_hash

    def _discretize(self, value: float, bins: int) -> int:
        return min(int(value * bins), bins - 1)

    def is_action_valid(self, action: CryptoAlgorithm, features: ContextFeatures,
                        security_level: SecurityLevel) -> bool:
        requirements = self.algorithm_requirements[action]

        if security_level.value < requirements['min_security_level'].value:
            return False

        if requirements['requires_qkd'] and not features.qkd_available:
            return False

        if requirements['quantum_hardware'] and not features.quantum_hardware_present:
            return False

        if features.available_resources < requirements['min_resources']:
            return False

        return True

    def get_valid_actions(self, features: ContextFeatures,
                          security_level: SecurityLevel) -> List[CryptoAlgorithm]:
        valid_actions = []
        for action in self.actions:
            if self.is_action_valid(action, features, security_level):
                valid_actions.append(action)

        if not valid_actions:
            valid_actions.append(CryptoAlgorithm.FALLBACK_AES)

        return valid_actions

    def compute_reward(self, action: CryptoAlgorithm, features: ContextFeatures,
                       security_level: SecurityLevel, outcome: Dict[str, Any]) -> float:
        
        lambda_1 = 8.0
        lambda_2 = 0.3
        lambda_3 = 0.2
        lambda_4 = 4.0
        lambda_5 = 3.0

        S_success = 1.0 if outcome.get("success", False) else 0.0

        T_latency = outcome.get("latency", 0.0) / 1000.0

        C_resource = self.algorithm_performance[action]['resource_cost']

        requirements = self.algorithm_requirements[action]
        security_match = (security_level.value >= requirements['min_security_level'].value)
        S_compliance = 1.0 if security_match else 0.0

        diversity_bonus = 0.0
        if action in [CryptoAlgorithm.QKD_BB84, CryptoAlgorithm.QKD_E91,
                      CryptoAlgorithm.QKD_CV, CryptoAlgorithm.QKD_MDI,
                      CryptoAlgorithm.QKD_DECOY]:
            diversity_bonus = 1.0
        elif action in [CryptoAlgorithm.PQC_KYBER, CryptoAlgorithm.PQC_DILITHIUM,
                        CryptoAlgorithm.PQC_NTRU, CryptoAlgorithm.PQC_SABER,
                        CryptoAlgorithm.PQC_FALCON]:
            diversity_bonus = 0.8
        elif action in [CryptoAlgorithm.HYBRID_RSA_PQC, CryptoAlgorithm.HYBRID_ECC_PQC]:
            diversity_bonus = 0.9
        elif action in [CryptoAlgorithm.RSA_4096, CryptoAlgorithm.ECC_521]:
            diversity_bonus = 0.5
        elif action == CryptoAlgorithm.AES_256_GCM:
            diversity_bonus = 0.3

        reward = (lambda_1 * S_success -
                  lambda_2 * T_latency -
                  lambda_3 * C_resource +
                  lambda_4 * S_compliance +
                  lambda_5 * diversity_bonus)

        if (security_level.value >= SecurityLevel.VERY_HIGH.value and
                features.qkd_available and
                action in [CryptoAlgorithm.QKD_BB84, CryptoAlgorithm.QKD_E91,
                           CryptoAlgorithm.QKD_CV, CryptoAlgorithm.QKD_MDI]):
            reward += 1.5

        if (security_level.value >= SecurityLevel.HIGH.value and
                not features.qkd_available and
                action in [CryptoAlgorithm.PQC_KYBER, CryptoAlgorithm.PQC_DILITHIUM,
                           CryptoAlgorithm.PQC_NTRU, CryptoAlgorithm.PQC_SABER]):
            reward += 1.5

        if (security_level.value >= SecurityLevel.VERY_HIGH.value and
                action in [CryptoAlgorithm.HYBRID_RSA_PQC, CryptoAlgorithm.HYBRID_ECC_PQC]):
            reward += 1.2

        return reward

    def update_algorithm_performance(self, action: CryptoAlgorithm,
                                     outcome: Dict[str, Any]):
        perf = self.algorithm_performance[action]

        alpha = 0.1

        if "latency" in outcome:
            perf['latency'] = (1 - alpha) * perf['latency'] + alpha * outcome['latency']

        if "success" in outcome:
            success = 1.0 if outcome['success'] else 0.0
            perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * success

        if "resource_usage" in outcome:
            perf['resource_cost'] = (1 - alpha) * perf['resource_cost'] + alpha * outcome['resource_usage']


def map_security_level(risk_score: float, conf_score: float) -> SecurityLevel:
    
    if risk_score is None:
        risk_score = 0.5
    if conf_score is None:
        conf_score = 0.5

    combined_score = max(risk_score, conf_score)

    if combined_score < 0.2:
        return SecurityLevel.VERY_LOW
    elif combined_score < 0.35:
        return SecurityLevel.LOW
    elif combined_score < 0.55:
        return SecurityLevel.MODERATE
    elif combined_score < 0.75:
        return SecurityLevel.HIGH
    elif combined_score < 0.9:
        return SecurityLevel.VERY_HIGH
    else:
        return SecurityLevel.ULTRA