package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.context_api.model.RiskContext;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Component
public class RiskServiceClient {

    private final WebClient webClient;

    public RiskServiceClient(@Qualifier("riskWebClient") WebClient riskWebClient) {
        this.webClient = riskWebClient;
    }

    public Optional<RiskContext> assessGeneral(String requestId, Map<String, Object> signals) {
        try {
            Map<String, Object> payload = new HashMap<>();
            if (requestId != null) payload.put("request_id", requestId);
            if (signals != null) payload.put("signals", signals);

            RiskContext resp = webClient.post()
                    .uri("/risk/assess")
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(RiskContext.class)
                    .timeout(Duration.ofMillis(1000))
                    .onErrorResume(e -> Mono.empty())
                    .block();
            return Optional.ofNullable(resp);
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    public Map<String, Object> buildBasicSignals(String geoRegion, String mfaStatus, String exposureLevel) {
        Map<String, Object> signals = new HashMap<>();
        if (geoRegion != null) signals.put("geo_region", geoRegion);
        if (mfaStatus != null) signals.put("mfa_status", mfaStatus);
        if (exposureLevel != null) signals.put("exposure_level", exposureLevel);
        // defaults exemplares
        signals.putIfAbsent("anomaly_index_global", 0.0);
        signals.putIfAbsent("incident_rate_7d", 0);
        signals.putIfAbsent("maintenance_window", false);
        return signals;
    }
}