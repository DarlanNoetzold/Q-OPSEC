package tech.noetzold.validation_send_api.model;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class NegotiationPayload {
    @NotBlank
    private String requestId;
    @NotBlank
    private String sessionId;

    private String selectedAlgorithm;

    @NotBlank
    private String cryptoNonceB64;
    @NotBlank
    private String cryptoCiphertextB64;
    @NotBlank
    private String cryptoAlgorithm;
    private Long cryptoExpiresAt;

    private String sourceId;
    @NotBlank
    private String originUrl;
}