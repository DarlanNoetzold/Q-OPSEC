package tech.noetzold.interceptor_api.model;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.*;

import java.time.Instant;

@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterceptedMessageResponse {
    private String request_id;
    private String message;
    private String source_id;
    private String destination_id;
    private JsonNode metadata;
    private String status;
    private Instant created_at;
    private Instant updated_at;

    public static InterceptedMessageResponse fromEntity(InterceptedMessage e) {
        return InterceptedMessageResponse.builder()
                .request_id(e.getRequestId())
                .message(e.getMessage())
                .source_id(e.getSourceId())
                .destination_id(e.getDestinationId())
                .metadata(e.getMetadataJson())
                .status(e.getStatus())
                .created_at(e.getCreatedAt())
                .updated_at(e.getUpdatedAt())
                .build();
    }
}