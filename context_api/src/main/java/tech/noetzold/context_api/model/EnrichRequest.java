package tech.noetzold.context_api.model;

import java.util.Map;

public record EnrichRequest(
        String requestId,
        Map<String,Object> source_hint,
        Map<String,Object> destination_hint,
        Map<String,String> headers,
        Map<String,Object> content_pointer
) {}