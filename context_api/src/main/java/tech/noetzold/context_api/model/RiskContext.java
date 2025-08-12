package tech.noetzold.context_api.model;

import java.util.List;
import java.util.Map;

public record RiskContext(
        Double score,
        String level,              // very_low, low, medium, high, critical
        Double anomaly_score,
        Map<String,Object> threat_intel,
        Integer recent_incidents,
        List<String> policy_overrides,
        String timestamp,
        String model_version
) {}
