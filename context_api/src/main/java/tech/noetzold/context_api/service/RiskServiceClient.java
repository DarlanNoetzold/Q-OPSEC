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

    private static final int MAX_RETRIES = 3;
    private static final Duration RETRY_DELAY = Duration.ofSeconds(2);
    private static final Duration TIMEOUT = Duration.ofMillis(1500);

    public RiskServiceClient(@Qualifier("riskWebClient") WebClient riskWebClient) {
        this.webClient = riskWebClient;
    }

    public Optional<RiskContext> assessGeneral(String requestId, Map<String, Object> signals) {
        try {
            Map<String, Object> payload = buildAssessPayload(requestId, signals);

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

    private Map<String, Object> buildAssessPayload(String requestId, Map<String, Object> signals) {
        Map<String, Object> payload = new HashMap<>();

        // request_id obrigatório (snake_case)
        payload.put("request_id", (requestId != null && !requestId.isBlank())
                ? requestId
                : "req_" + System.currentTimeMillis());

        // signals: garantir objeto não-nulo e saneado
        Map<String, Object> s = new HashMap<>();
        if (signals != null) {
            for (Map.Entry<String, Object> e : signals.entrySet()) {
                if (e.getKey() == null) continue;
                Object v = e.getValue();
                if (v == null) continue;
                // permite apenas tipos simples comuns ao schema
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
                            // 503 -> modelo não pronto
                            return Mono.just(new AssessOutcome(Optional.empty(), true));
                        } else if (status == HttpStatus.BAD_REQUEST) {
                            // 400 -> loga corpo para identificar exatamente o campo inválido
                            return resp.bodyToMono(String.class)
                                    .doOnNext(body -> System.err.println("Risk 400 payload error: " + body))
                                    .then(Mono.just(new AssessOutcome(Optional.empty(), false)));
                        } else {
                            return Mono.just(new AssessOutcome(Optional.empty(), false));
                        }
                    })
                    .timeout(TIMEOUT)
                    .onErrorResume(e -> {
                        System.err.println("Risk assess error: " + e.getMessage());
                        return Mono.just(new AssessOutcome(Optional.empty(), false));
                    })
                    .blockOptional()
                    .orElse(new AssessOutcome(Optional.empty(), false));
        } catch (Exception e) {
            System.err.println("Risk assess exception: " + e.getMessage());
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