package tech.noetzold.context_api.model;

public record EnrichResponse(
        String request_id,
        SourceContext source,
        DestinationContext destination,
        RiskContext risk,
        ContentConfidentiality confidentiality
) {}
