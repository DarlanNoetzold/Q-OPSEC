package tech.noetzold.context_api.model;

public record SourceContext(
        String ip,
        String user_id,
        String device_id,
        String user_agent,
        String geo,
        String os_version,
        String device_type,
        String mfa_status,
        String security_status
) {}
