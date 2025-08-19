package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.RiskContext;
import tech.noetzold.context_api.model.SourceContext;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
public class RiskAssessmentFacade {

    private final RiskServiceClient riskClient;

    public RiskAssessmentFacade(RiskServiceClient riskClient) {
        this.riskClient = riskClient;
    }

    public RiskContext getRisk(String reqId, SourceContext src, DestinationContext dest) {
        String rid = (reqId == null || reqId.isBlank()) ? java.util.UUID.randomUUID().toString() : reqId;

        Map<String, Object> signals = new HashMap<>();
        if (src != null) {
            if (src.geo() != null) signals.put("geo_region", src.geo());
            if (src.mfa_status() != null) signals.put("mfa_status", src.mfa_status());
        }
        if (dest != null) {
            String exposure = deriveExposureFromPolicy(dest.security_policy());
            if (exposure != null) signals.put("exposure_level", exposure);
        }

        signals.putIfAbsent("global_alert_level", "low");
        signals.putIfAbsent("current_campaigns", java.util.List.of());
        signals.putIfAbsent("anomaly_index_global", 0.0);
        signals.putIfAbsent("incident_rate_7d", 0);
        signals.putIfAbsent("patch_delay_days_p50", 0);
        signals.putIfAbsent("exposure_level", "medium");
        signals.putIfAbsent("maintenance_window", false);
        signals.putIfAbsent("compliance_debt_score", 0.0);
        signals.putIfAbsent("business_critical_period", false);

        return riskClient.assessGeneral(rid, signals).orElseGet(() -> {
            Map<String, Object> meta = new HashMap<>();
            meta.put("source", "fallback");
            return new RiskContext(
                    0.3, "low", 0.1,
                    meta, 0, java.util.List.of(),
                    java.time.Instant.now().toString(),
                    "risk-fb-0.0.1"
            );
        });
    }

    private String deriveExposureFromPolicy(String policy) {
        if (policy == null) return null;
        String p = policy.trim().toLowerCase();
        if (p.contains("strict") || p.contains("high")) return "high";
        if (p.contains("medium") || p.contains("moderate")) return "medium";
        if (p.contains("low") || p.contains("permissive")) return "low";
        return null;
    }
}