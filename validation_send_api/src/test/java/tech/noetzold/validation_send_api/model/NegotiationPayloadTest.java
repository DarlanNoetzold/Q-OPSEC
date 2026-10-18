package tech.noetzold.validation_send_api.model;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class NegotiationPayloadTest {
    private final Validator validator = Validation.buildDefaultValidatorFactory().getValidator();

    @Test
    void rejectsPayloadWhenRequiredFieldsAreBlank() {
        NegotiationPayload payload = new NegotiationPayload();

        assertThat(validator.validate(payload))
                .extracting(violation -> violation.getPropertyPath().toString())
                .containsExactlyInAnyOrder("requestId", "sessionId", "cryptoNonceB64",
                        "cryptoCiphertextB64", "cryptoAlgorithm", "originUrl");
    }

    @Test
    void acceptsPayloadWhenAllRequiredFieldsArePresent() {
        NegotiationPayload payload = new NegotiationPayload();
        payload.setRequestId("request-1");
        payload.setSessionId("session-1");
        payload.setCryptoNonceB64("nonce");
        payload.setCryptoCiphertextB64("ciphertext");
        payload.setCryptoAlgorithm("AES-GCM");
        payload.setOriginUrl("http://localhost:8080/origin");

        assertThat(validator.validate(payload)).isEmpty();
    }
}
