package tech.noetzold.interceptor_api.model;

import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ContextResponse {
    private String source;
    private String destination;
    private Map<String, Object> risk;
    private Map<String, Object> confidentiality;
}
