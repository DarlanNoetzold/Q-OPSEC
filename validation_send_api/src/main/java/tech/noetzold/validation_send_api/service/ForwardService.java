package tech.noetzold.validation_send_api.service;


import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.model.NegotiationPayload;
import tech.noetzold.validation_send_api.repository.EncryptedDeliveryRepository;

import java.time.Duration;
import java.util.function.Function;

@Service
@RequiredArgsConstructor
public class ForwardService {

    private final EncryptedDeliveryRepository repository;
    private final WebClient webClient = WebClient.builder().build();

    @Transactional
    public EncryptedDelivery receiveAndPersist(NegotiationPayload p) {
        // cria/atualiza registro como pending
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

    @Transactional
    public EncryptedDelivery markDelivered(EncryptedDelivery entity) {
        entity.setStatus("delivered");
        entity.setErrorMessage(null);
        return repository.save(entity);
    }

    @Transactional
    public EncryptedDelivery markFailed(EncryptedDelivery entity, String error) {
        entity.setStatus("failed");
        entity.setErrorMessage(error != null ? error.substring(0, Math.min(2000, error.length())) : null);
        return repository.save(entity);
    }

    public void forwardToOriginOrThrow(NegotiationPayload payload) {
        if (payload.getOriginUrl() == null || payload.getOriginUrl().isBlank()) {
            throw new IllegalArgumentException("originUrl not provided");
        }

        try {
            webClient.post()
                    .uri(payload.getOriginUrl())
                    .bodyValue(payload)
                    .retrieve()
                    .toBodilessEntity()
                    .timeout(Duration.ofSeconds(10))  // timeout de 10s
                    .block();
        } catch (Exception e) {
            throw new RuntimeException("Failed to forward to origin: " + e.getMessage(), e);
        }
    }
}