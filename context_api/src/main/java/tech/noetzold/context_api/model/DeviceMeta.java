package tech.noetzold.context_api.model;

public record DeviceMeta(
        String osVersion,
        String deviceType,
        String mfaStatus,
        String securityStatus
) {}
