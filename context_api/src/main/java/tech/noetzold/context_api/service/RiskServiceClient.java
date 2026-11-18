package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;
import tech.noetzold.context_api.model.RiskContext;
import tech.noetzold.context_api.resilience.CircuitBreaker;
import tech.noetzold.context_api.resilience.CircuitOpenException;
import tech.noetzold.context_api.resilience.UpstreamUnavailableException;

import java.io.IOException;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeoutException;

@Component
public class RiskServiceClient {

    private final WebClient webClient;
    private final CircuitBreaker breaker =
            new CircuitBreaker("risk_service", 5, Duration.ofSeconds(30), 1);

    private static final int MAX_RETRIES = 3;
    private static final Duration RETRY_DELAY = Duration.ofMillis(500);
    private static final Duration MAX_BACKOFF = Duration.ofSeconds(5);
    private static final Duration TIMEOUT = Duration.ofMillis(1500);

    public RiskServiceClient(@Qualifier("riskWebClient") WebClient riskWebClient) {
        this.webClient = riskWebClient;
    }

    public Optional<RiskContext> assessGeneral(String requestId, Map<String, Object> signals) {
        Map<String, Object> payload = buildAssessPayload(requestId, signals);

        try {
            breaker.acquire();
        } catch (CircuitOpenException ex) {
            System.err.println("Risk circuit open: " + ex.getMessage());
            return Optional.empty();
        }

        try {
            Optional<RiskContext> result = assessWithWarmup(payload);
            breaker.recordSuccess();
            return result;
        } catch (Exception ex) {
            breaker.recordFailure(ex);
            System.err.println("Risk assess failed: " + ex.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Assesses once; when the model reports "not ready" it trains the default
     * dataset and retries with a reactive backoff instead of sleeping the thread.
     */
    private Optional<RiskContext> assessWithWarmup(Map<String, Object> payload) {
        return doAssess(payload)
                .flatMap(first -> {
                    if (first.context().isPresent() || !first.modelNotReady()) {
                        return Mono.just(first);
                    }
                    return Mono.fromCallable(this::tryTrainDefault)
                            .thenMany(Flux.range(1, MAX_RETRIES)
                                    .concatMap(attempt -> Mono.delay(RETRY_DELAY.multipliedBy(attempt))
                                            .then(doAssess(payload))))
                            .filter(outcome -> outcome.context().isPresent())
                            .next()
                            .switchIfEmpty(Mono.just(new AssessOutcome(Optional.empty(), false)));
                })
                .blockOptional()
                .orElse(new AssessOutcome(Optional.empty(), false))
                .context();
    }

    private Map<String, Object> buildAssessPayload(String requestId, Map<String, Object> signals) {
        Map<String, Object> payload = new HashMap<>();

        payload.put("request_id", (requestId != null && !requestId.isBlank())
                ? requestId
                : "req_" + System.currentTimeMillis());

        Map<String, Object> s = new HashMap<>();
        if (signals != null) {
            for (Map.Entry<String, Object> e : signals.entrySet()) {
                if (e.getKey() == null) continue;
                Object v = e.getValue();
                if (v == null) continue;
                if (v instanceof String || v instanceof Number || v instanceof Boolean) {
                    s.put(toSnakeCase(e.getKey()), v);
                }
            }
        }
        if (s.isEmpty()) {
            s.put("global_alert_level", "low");
        }

        payload.put("signals", s);
        return payload;
    }

    private Mono<AssessOutcome> doAssess(Map<String, Object> payload) {
        return webClient.post()
                .uri("/risk/assess")
                .bodyValue(payload)
                .exchangeToMono(resp -> {
                    HttpStatusCode status = resp.statusCode();
                    if (status.is2xxSuccessful()) {
                        return resp.bodyToMono(RiskContext.class)
                                .map(rc -> new AssessOutcome(Optional.ofNullable(rc), false));
                    }
                    if (status == HttpStatus.SERVICE_UNAVAILABLE) {
                        return Mono.just(new AssessOutcome(Optional.empty(), true));
                    }
                    if (status == HttpStatus.BAD_REQUEST) {
                        return resp.bodyToMono(String.class)
                                .doOnNext(body -> System.err.println("Risk 400 payload error: " + body))
                                .then(Mono.just(new AssessOutcome(Optional.empty(), false)));
                    }
                    return Mono.error(new UpstreamUnavailableException("risk service status " + status.value()));
                })
                .timeout(TIMEOUT)
                .retryWhen(Retry.backoff(MAX_RETRIES, RETRY_DELAY)
                        .maxBackoff(MAX_BACKOFF)
                        .jitter(0.3)
                        .filter(RiskServiceClient::isRetryable)
                        .onRetryExhaustedThrow((spec, signal) -> signal.failure()));
    }

    private static boolean isRetryable(Throwable error) {
        return error instanceof UpstreamUnavailableException
                || error instanceof TimeoutException
                || error instanceof IOException;
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

    private String toSnakeCase(String key) {
        StringBuilder sb = new StringBuilder();
        for (char c : key.toCharArray()) {
            if (Character.isUpperCase(c)) {
                sb.append('_').append(Character.toLowerCase(c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString().replaceAll("__+", "_");
    }

    private record AssessOutcome(Optional<RiskContext> context, boolean modelNotReady) {}
}
