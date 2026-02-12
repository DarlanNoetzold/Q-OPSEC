package tech.noetzold.context_api.model;

import java.util.List;
import java.util.Map;

public record ContentConfidentiality(
        String classification,  // public, internal, confidential, restricted
        Double score,
        List<String> tags,
        List<String> detected_patterns,
        List<Map<String,Object>> dlp_findings,
        String source_app_context,
        String user_label,
        String model_version
) {}
