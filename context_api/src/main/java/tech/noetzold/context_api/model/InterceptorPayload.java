package tech.noetzold.context_api.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public record InterceptorPayload(
        @JsonProperty("request_id")
        String requestId,
        String sourceId,
        String destinationId,
        String content,
        Map<String, Object> metadata
) {}