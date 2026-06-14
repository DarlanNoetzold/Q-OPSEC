package tech.noetzold.interceptor_api.model;

import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterceptRequest {
    private String request_id;
    private String requestId;
    private String message;
    private String sourceId;
    private String destinationId;
    private Map<String, Object> metadata;

    public String getAnyRequestId() {
        return request_id != null ? request_id : requestId;
    }
}
