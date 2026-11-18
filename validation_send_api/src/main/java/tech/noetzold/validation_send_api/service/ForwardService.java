package tech.noetzold.validation_send_api.service;


import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import reactor.util.retry.Retry;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.model.NegotiationPayload;
import tech.noetzold.validation_send_api.repository.EncryptedDeliveryRepository;
import tech.noetzold.validation_send_api.resilience.CircuitBreaker;
import tech.noetzold.validation_send_api.resilience.CircuitOpenException;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.TimeoutException;

@Service
@RequiredArgsConstructor
public class ForwardService {

    private static final int MAX_ATTEMPTS = 3;
    private static final Duration RETRY_DELAY = Duration.ofMillis(500);
    private static final Duration MAX_BACKOFF = Duration.ofSeconds(5);
    private static final Duration TIMEOUT = Duration.ofSeconds(10);

    private final EncryptedDeliveryRepository repository;
    private final WebClient webClient = WebClient.builder().build();
    private final CircuitBreaker originBreaker =
            new CircuitBreaker("origin_callback", 5, Duration.ofSeconds(30), 1);

    @Transactional
    public EncryptedDelivery receiveAndPersist(NegotiationPayload p) {
        EncryptedDelivery entity = repository.findByRequestId(p.getRequestId())
                .orElseGet(EncryptedDelivery::new);

        entity.setRequestId(p.getRequestId());
        entity.setSessionId(p.getSessionId());
        entity.setSelectedAlgorithm(p.getSelectedAlgorithm());
        entity.setCryptoAlgorithm(p.getCryptoAlgorithm());
        entity.setCryptoNonceB64(p.getCryptoNonceB64());
        entity.setCryptoCiphertextB64(p.getCryptoCiphertextB64());
        entity.setCryptoExpiresAt(p.getCryptoExpiresAt());
        entity.setSourceId(p.getSourceId());
        entity.setOriginUrl(p.getOriginUrl());
        entity.setStatus("pending");
        entity.setErrorMessage(null);

        return repository.save(entity);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public EncryptedDelivery markDelivered(EncryptedDelivery entity) {
        entity.setStatus("delivered");
        entity.setErrorMessage(null);
        return repository.save(entity);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public EncryptedDelivery markFailed(EncryptedDelivery entity, String error) {
        entity.setStatus("failed");
        entity.setErrorMessage(error != null ? error.substring(0, Math.min(2000, error.length())) : null);
        return repository.save(entity);
    }

    @Transactional(propagation = Propagation.NOT_SUPPORTED)
    public void forwardToOriginOrThrow(NegotiationPayload payload) {
        if (payload.getOriginUrl() == null || payload.getOriginUrl().isBlank()) {
            throw new IllegalArgumentException("originUrl not provided");
        }

        try {
            originBreaker.acquire();
        } catch (CircuitOpenException ex) {
            throw new RuntimeException("Origin callback circuit open: " + ex.getMessage(), ex);
        }

        try {
            webClient.post()
                    .uri(payload.getOriginUrl())
                    .bodyValue(payload)
                    .retrieve()
                    .toBodilessEntity()
                    .timeout(TIMEOUT)
                    .retryWhen(Retry.backoff(MAX_ATTEMPTS, RETRY_DELAY)
                            .maxBackoff(MAX_BACKOFF)
                            .jitter(0.3)
                            .filter(ForwardService::isRetryable)
                            .onRetryExhaustedThrow((spec, signal) -> signal.failure()))
                    .block();
            originBreaker.recordSuccess();
        } catch (Exception e) {
            originBreaker.recordFailure(e);
            throw new RuntimeException("Failed to forward to origin: " + e.getMessage(), e);
        }
    }

    private static boolean isRetryable(Throwable error) {
        if (error instanceof WebClientResponseException responseException) {
            return responseException.getStatusCode().is5xxServerError();
        }
        return error instanceof TimeoutException || error instanceof IOException;
    }
}