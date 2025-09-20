package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
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

    // Configurações de retry
    private static final int MAX_RETRIES = 3;                 // até 3 tentativas após treinar
    private static final Duration RETRY_DELAY = Duration.ofSeconds(2); // espera entre tentativas
    private static final Duration TIMEOUT = Duration.ofMillis(1000);   // timeout das chamadas

    public RiskServiceClient(@Qualifier("riskWebClient") WebClient riskWebClient) {
        this.webClient = riskWebClient;
    }

    public Optional<RiskContext> assessGeneral(String requestId, Map<String, Object> signals) {
        try {
            Map<String, Object> payload = new HashMap<>();
            if (requestId != null) payload.put("requestId", requestId);
            if (signals != null) payload.put("signals", signals);

            AssessOutcome first = doAssess(payload);
            if (first.context().isPresent()) {
                return first.context();
            }

            if (first.modelNotReady()) {
                boolean trained = tryTrainDefault();
                for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                    sleepQuietly(RETRY_DELAY.multipliedBy(attempt)); // backoff linear
                    AssessOutcome retry = doAssess(payload);
                    if (retry.context().isPresent()) {
                        return retry.context();
                    }
                }
            }

            return Optional.empty();

        } catch (Exception e) {
            return Optional.empty();
        }
    }

    private AssessOutcome doAssess(Map<String, Object> payload) {
        try {
            return webClient.post()
                    .uri("/risk/assess")
                    .bodyValue(payload)
                    .exchangeToMono(resp -> {
                        HttpStatusCode status = resp.statusCode();
                        if (status.is2xxSuccessful()) {
                            return resp.bodyToMono(RiskContext.class)
                                    .map(rc -> new AssessOutcome(Optional.ofNullable(rc), false));
                        } else if (status == HttpStatus.SERVICE_UNAVAILABLE) {
                            return Mono.just(new AssessOutcome(Optional.empty(), true));
                        } else {
                            return Mono.just(new AssessOutcome(Optional.empty(), false));
                        }
                    })
                    .timeout(TIMEOUT)
                    .onErrorResume(e -> Mono.just(new AssessOutcome(Optional.empty(), false)))
                    .blockOptional()
                    .orElse(new AssessOutcome(Optional.empty(), false));
        } catch (Exception e) {
            return new AssessOutcome(Optional.empty(), false);
        }
    }

    private boolean tryTrainDefault() {
        try {
            Map<String, Object> trainPayload = Map.of("dataset_id", "default");
            return webClient.post()
                    .uri("/risk/train")
                    .bodyValue(trainPayload)
                    .exchangeToMono(resp -> Mono.just(resp.statusCode().is2xxSuccessful()))
                    .timeout(TIMEOUT)
                    .onErrorResume(e -> Mono.just(false))
                    .blockOptional()
                    .orElse(false);
        } catch (Exception e) {
            return false;
        }
    }

    private void sleepQuietly(Duration d) {
        try {
            Thread.sleep(Math.max(1, d.toMillis()));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    private record AssessOutcome(Optional<RiskContext> context, boolean modelNotReady) {}

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