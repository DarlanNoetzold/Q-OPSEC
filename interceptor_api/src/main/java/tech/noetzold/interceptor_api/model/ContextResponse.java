package tech.noetzold.interceptor_api.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
@JsonIgnoreProperties(ignoreUnknown = true)
public class ContextResponse {

    @JsonProperty("request_id")
    private String requestId;

    private Map<String, Object> source;
    private Map<String, Object> destination;

    // Você já usava Map, está ok manter como Map
    private Map<String, Object> risk;
    private Map<String, Object> confidentiality;
}