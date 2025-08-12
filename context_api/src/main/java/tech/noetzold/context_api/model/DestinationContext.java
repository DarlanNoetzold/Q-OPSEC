package tech.noetzold.context_api.model;

import java.util.List;

public record DestinationContext(
        String ip,
        String service_id,
        String service_type,
        String security_policy,
        String security_status,
        String os_version,
        List<String> allowed_protocols
) {}
