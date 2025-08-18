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
        // Monta signals a partir do contexto disponível
        Map<String, Object> signals = new HashMap<>();
        if (src != null) {
            if (src.geo() != null) signals.put("geo_region", src.geo());
            if (src.mfaStatus() != null) signals.put("mfa_status", src.mfaStatus());
        }
        if (dest != null) {
            String exposure = deriveExposureFromPolicy(dest.securityPolicy());
            if (exposure != null) signals.put("exposure_level", exposure);
        }
        // Defaults defensivos (ajuste conforme o seu modelo de features)
        signals.putIfAbsent("anomaly_index_global", 0.0);
        signals.putIfAbsent("incident_rate_7d", 0);
        signals.putIfAbsent("maintenance_window", false);

        return riskClient.assessGeneral(reqId, signals).orElseGet(() -> {
            Map<String, Object> meta = new HashMap<>();
            meta.put("source", "fallback");
            return new RiskContext(
                    0.3,                       // score
                    "low",                      // level
                    0.1,                        // confidence
                    meta,                       // metadata (HashMap para evitar NPE de Map.of com null)
                    0,                          // model_version or flags, conforme seu POJO
                    List.of(),                  // contributors / rules acionadas
                    Instant.now().toString(),   // timestamp
                    "risk-fb-0.0.1"             // engine/version
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