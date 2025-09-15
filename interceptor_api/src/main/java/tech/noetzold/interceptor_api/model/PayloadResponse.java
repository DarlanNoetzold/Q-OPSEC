package tech.noetzold.interceptor_api.model;

import lombok.*;

import java.util.Base64;
import java.util.Map;

@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class PayloadResponse {
    private String request_id;
    private String payload_b64;
    private String content_type;
    private Map<String, Object> metadata;

    public static PayloadResponse fromEntity(InterceptedMessage e) {
        String payloadB64 = e.getMessage() != null
                ? Base64.getEncoder().encodeToString(e.getMessage().getBytes())
                : null;

        return PayloadResponse.builder()
                .request_id(e.getRequestId())
                .payload_b64(payloadB64)
                .content_type("application/json")
                .metadata(e.getMetadataJson() == null ? null : Map.of("raw", e.getMetadataJson()))
                .build();
    }
}