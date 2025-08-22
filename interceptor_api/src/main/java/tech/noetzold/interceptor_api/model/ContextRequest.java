package tech.noetzold.interceptor_api.model;

import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ContextRequest {
    private String sourceId;
    private String destinationId;
    private String content;
    private Map<String, Object> metadata;
}
