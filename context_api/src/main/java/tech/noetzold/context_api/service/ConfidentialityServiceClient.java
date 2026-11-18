package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.util.retry.Retry;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;
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
public class ConfidentialityServiceClient {
    private final WebClient webClient;
    private final CircuitBreaker breaker =
            new CircuitBreaker("confiability_service", 5, Duration.ofSeconds(30), 1);

    private static final int MAX_RETRIES = 3;
    private static final Duration RETRY_DELAY = Duration.ofMillis(500);
    private static final Duration MAX_BACKOFF = Duration.ofSeconds(5);
    private static final Duration TIMEOUT = Duration.ofMillis(1500);

    public ConfidentialityServiceClient(@Qualifier("confWebClient") WebClient confWebClient) {
        this.webClient = confWebClient;
    }

    public Optional<ContentConfidentiality> classify(String requestId,
                                                     Map<String, Object> contentPointer,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        Map<String, Object> payload = buildClassifyPayload(requestId, contentPointer, src, dst);

        try {
            breaker.acquire();
        } catch (CircuitOpenException ex) {
            System.err.println("Confidentiality circuit open: " + ex.getMessage());
            return Optional.empty();
        }

        try {
            Optional<ContentConfidentiality> result = classifyWithWarmup(payload);
            breaker.recordSuccess();
            return result;
        } catch (Exception ex) {
            breaker.recordFailure(ex);
            System.err.println("Confidentiality classify failed: " + ex.getMessage());
            return Optional.empty();
        }
    }

    private Optional<ContentConfidentiality> classifyWithWarmup(Map<String, Object> payload) {
        return doClassify(payload)
                .flatMap(first -> {
                    if (first.result().isPresent() || !first.modelNotReady()) {
                        return Mono.just(first);
                    }
                    return Mono.fromCallable(this::tryTrainDefault)
                            .thenMany(Flux.range(1, MAX_RETRIES)
                                    .concatMap(attempt -> Mono.delay(RETRY_DELAY.multipliedBy(attempt))
                                            .then(doClassify(payload))))
                            .filter(outcome -> outcome.result().isPresent())
                            .next()
                            .switchIfEmpty(Mono.just(new ClassifyOutcome(Optional.empty(), false)));
                })
                .blockOptional()
                .orElse(new ClassifyOutcome(Optional.empty(), false))
                .result();
    }

    private Map<String, Object> buildClassifyPayload(String requestId,
                                                     Map<String, Object> contentPointer,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        Map<String, Object> body = new HashMap<>();

        body.put("payload", contentPointer != null ? contentPointer : new HashMap<>());

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("source_id", "context_api_phd_v2");
        metadata.put("entity_id", src != null ? src.ip() : "192.168.18.18");
        metadata.put("request_id", requestId);

        body.put("metadata", metadata);
        return body;
    }

    private Mono<ClassifyOutcome> doClassify(Map<String, Object> payload) {
        return webClient.post()
                .uri("/api/v2/trust/evaluate")
                .bodyValue(payload)
                .exchangeToMono(resp -> {
                    HttpStatusCode status = resp.statusCode();
                    if (status.is2xxSuccessful()) {
                        return resp.bodyToMono(ContentConfidentiality.class)
                                .map(cc -> new ClassifyOutcome(Optional.ofNullable(cc), false));
                    }
                    if (status == HttpStatus.SERVICE_UNAVAILABLE) {
                        return Mono.just(new ClassifyOutcome(Optional.empty(), true));
                    }
                    if (status == HttpStatus.BAD_REQUEST) {
                        return resp.bodyToMono(String.class)
                                .doOnNext(body -> System.err.println("Confidentiality 400: " + body))
                                .then(Mono.just(new ClassifyOutcome(Optional.empty(), false)));
                    }
                    return Mono.error(new UpstreamUnavailableException(
                            "confidentiality service status " + status.value()));
                })
                .timeout(TIMEOUT)
                .retryWhen(Retry.backoff(MAX_RETRIES, RETRY_DELAY)
                        .maxBackoff(MAX_BACKOFF)
                        .jitter(0.3)
                        .filter(ConfidentialityServiceClient::isRetryable)
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
                    .uri("/confidentiality/train")
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

    private record ClassifyOutcome(Optional<ContentConfidentiality> result, boolean modelNotReady) {}
}
