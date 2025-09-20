package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Component
public class ConfidentialityServiceClient {
    private final WebClient webClient;

    private static final int MAX_RETRIES = 3;
    private static final Duration RETRY_DELAY = Duration.ofSeconds(2);
    private static final Duration TIMEOUT = Duration.ofMillis(1500);

    public ConfidentialityServiceClient(@Qualifier("confWebClient") WebClient confWebClient) {
        this.webClient = confWebClient;
    }

    public Optional<ContentConfidentiality> classify(String requestId,
                                                     Map<String, Object> contentPointer,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        try {
            Map<String, Object> payload = buildClassifyPayload(requestId, contentPointer, src, dst);

            ClassifyOutcome first = doClassify(payload);
            if (first.result().isPresent()) {
                return first.result();
            }

            if (first.modelNotReady()) {
                boolean trained = tryTrainDefault();
                for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
                    sleepQuietly(RETRY_DELAY.multipliedBy(attempt)); // backoff linear
                    ClassifyOutcome retry = doClassify(payload);
                    if (retry.result().isPresent()) {
                        return retry.result();
                    }
                }
            }

            return Optional.empty();

        } catch (Exception e) {
            return Optional.empty();
        }
    }

    private Map<String, Object> buildClassifyPayload(String requestId,
                                                     Map<String, Object> contentPointer,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        Map<String, Object> payload = new HashMap<>();

        payload.put("request_id", (requestId != null && !requestId.isBlank())
                ? requestId
                : "req_" + System.currentTimeMillis());

        Map<String, Object> cp = new HashMap<>();
        if (contentPointer != null && !contentPointer.isEmpty()) {
            if (contentPointer.get("ref") != null) cp.put("ref", contentPointer.get("ref"));
            if (contentPointer.get("sample_text") != null) cp.put("sample_text", contentPointer.get("sample_text"));
            if (contentPointer.get("metadata") != null) cp.put("metadata", contentPointer.get("metadata"));
        }
        payload.put("content_pointer", cp);

        if (src != null) {
            Map<String, Object> sourceData = new HashMap<>();
            if (src.ip() != null) sourceData.put("ip", src.ip());
            if (src.user_id() != null) sourceData.put("user_id", src.user_id());
            if (!sourceData.isEmpty()) {
                payload.put("source", sourceData);
            }
        }

        if (dst != null && dst.service_id() != null) {
            Map<String, Object> destData = new HashMap<>();
            destData.put("service_id", dst.service_id());
            payload.put("destination", destData);
        }

        return payload;
    }

    private ClassifyOutcome doClassify(Map<String, Object> payload) {
        try {
            return webClient.post()
                    .uri("/confidentiality/classify")
                    .bodyValue(payload)
                    .exchangeToMono(resp -> {
                        HttpStatusCode status = resp.statusCode();
                        if (status.is2xxSuccessful()) {
                            return resp.bodyToMono(ContentConfidentiality.class)
                                    .map(cc -> new ClassifyOutcome(Optional.ofNullable(cc), false));
                        } else if (status == HttpStatus.SERVICE_UNAVAILABLE) {
                            return Mono.just(new ClassifyOutcome(Optional.empty(), true));
                        } else if (status == HttpStatus.BAD_REQUEST) {
                            return resp.bodyToMono(String.class)
                                    .doOnNext(body -> System.err.println("Confidentiality 400: " + body))
                                    .then(Mono.just(new ClassifyOutcome(Optional.empty(), false)));
                        } else {
                            return Mono.just(new ClassifyOutcome(Optional.empty(), false));
                        }
                    })
                    .timeout(TIMEOUT)
                    .onErrorResume(e -> {
                        System.err.println("Confidentiality classify error: " + e.getMessage());
                        return Mono.just(new ClassifyOutcome(Optional.empty(), false));
                    })
                    .blockOptional()
                    .orElse(new ClassifyOutcome(Optional.empty(), false));
        } catch (Exception e) {
            System.err.println("Confidentiality classify exception: " + e.getMessage());
            return new ClassifyOutcome(Optional.empty(), false);
        }
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

    private void sleepQuietly(Duration d) {
        try {
            Thread.sleep(Math.max(1, d.toMillis()));
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
    }

    private record ClassifyOutcome(Optional<ContentConfidentiality> result, boolean modelNotReady) {}
}