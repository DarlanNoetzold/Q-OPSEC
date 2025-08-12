package tech.noetzold.context_api.model;

import java.util.List;

public record ServiceMeta(
        String ip,
        String serviceType,
        String securityPolicy,
        String securityStatus,
        String osVersion,
        List<String> allowedProtocols
) {}