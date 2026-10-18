package tech.noetzold.validation_send_api.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.model.NegotiationPayload;
import tech.noetzold.validation_send_api.repository.EncryptedDeliveryRepository;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ForwardServiceTest {
    @Mock
    private EncryptedDeliveryRepository repository;

    private ForwardService service;

    @BeforeEach
    void setUp() {
        service = new ForwardService(repository);
    }

    @Test
    void persistsNewDeliveryAsPendingWithPayloadFields() {
        NegotiationPayload payload = payload();
        when(repository.findByRequestId("request-1")).thenReturn(Optional.empty());
        when(repository.save(any(EncryptedDelivery.class))).thenAnswer(invocation -> invocation.getArgument(0));

        EncryptedDelivery saved = service.receiveAndPersist(payload);

        assertThat(saved.getRequestId()).isEqualTo("request-1");
        assertThat(saved.getSessionId()).isEqualTo("session-1");
        assertThat(saved.getSelectedAlgorithm()).isEqualTo("X25519");
        assertThat(saved.getCryptoAlgorithm()).isEqualTo("AES-GCM");
        assertThat(saved.getCryptoNonceB64()).isEqualTo("nonce");
        assertThat(saved.getCryptoCiphertextB64()).isEqualTo("ciphertext");
        assertThat(saved.getCryptoExpiresAt()).isEqualTo(123456L);
        assertThat(saved.getSourceId()).isEqualTo("source-1");
        assertThat(saved.getOriginUrl()).isEqualTo("http://localhost:8080/origin");
        assertThat(saved.getStatus()).isEqualTo("pending");
        assertThat(saved.getErrorMessage()).isNull();
    }

    @Test
    void refreshesExistingDeliveryInsteadOfCreatingAnother() {
        NegotiationPayload payload = payload();
        EncryptedDelivery existing = EncryptedDelivery.builder()
                .id(42L).requestId("request-1").status("failed").errorMessage("old error").build();
        when(repository.findByRequestId("request-1")).thenReturn(Optional.of(existing));
        when(repository.save(any(EncryptedDelivery.class))).thenAnswer(invocation -> invocation.getArgument(0));

        EncryptedDelivery saved = service.receiveAndPersist(payload);

        assertThat(saved).isSameAs(existing);
        assertThat(saved.getId()).isEqualTo(42L);
        assertThat(saved.getStatus()).isEqualTo("pending");
        assertThat(saved.getErrorMessage()).isNull();
        verify(repository).save(existing);
    }

    @Test
    void truncatesFailureReasonToDatabaseSafeLimit() {
        EncryptedDelivery delivery = EncryptedDelivery.builder().requestId("request-1").build();
        String longError = "x".repeat(2_100);
        when(repository.save(delivery)).thenReturn(delivery);

        EncryptedDelivery saved = service.markFailed(delivery, longError);

        assertThat(saved.getStatus()).isEqualTo("failed");
        assertThat(saved.getErrorMessage()).hasSize(2_000);
    }

    @Test
    void clearsFailureReasonWhenDeliveryIsMarkedDelivered() {
        EncryptedDelivery delivery = EncryptedDelivery.builder()
                .requestId("request-1").status("failed").errorMessage("previous error").build();
        when(repository.save(delivery)).thenReturn(delivery);

        EncryptedDelivery saved = service.markDelivered(delivery);

        assertThat(saved.getStatus()).isEqualTo("delivered");
        assertThat(saved.getErrorMessage()).isNull();
    }

    private static NegotiationPayload payload() {
        NegotiationPayload payload = new NegotiationPayload();
        payload.setRequestId("request-1");
        payload.setSessionId("session-1");
        payload.setSelectedAlgorithm("X25519");
        payload.setCryptoAlgorithm("AES-GCM");
        payload.setCryptoNonceB64("nonce");
        payload.setCryptoCiphertextB64("ciphertext");
        payload.setCryptoExpiresAt(123456L);
        payload.setSourceId("source-1");
        payload.setOriginUrl("http://localhost:8080/origin");
        return payload;
    }
}
