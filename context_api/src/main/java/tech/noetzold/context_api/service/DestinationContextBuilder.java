package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.ServiceRepository;

import java.util.List;
import java.util.Map;
import java.util.Optional;

@Component
public class DestinationContextBuilder {

    private final ServiceRepository serviceRepo;

    public DestinationContextBuilder(ServiceRepository serviceRepo) {
        this.serviceRepo = serviceRepo;
    }

    public DestinationContext build(Map<String,Object> destHint, EnrichRequest req) {
        var dh = Optional.ofNullable(destHint).orElse(Map.of());
        String destHost = (String) dh.getOrDefault("host", getHeader(req, "X-Dest-Host"));
        String serviceId = (String) dh.getOrDefault("service_id", getHeader(req, "X-Service-Id"));
        String path = (String) dh.getOrDefault("path", getHeader(req, "X-Request-Path"));

        var svc = serviceRepo.findByServiceIdAndHost(serviceId, destHost, path)
                .orElse(new ServiceMeta("0.0.0.0","unknown","medium","unknown","unknown",
                        List.of("TLS1.2","TLS1.3")));

        return new DestinationContext(
                svc.ip(), serviceId, svc.serviceType(),
                svc.securityPolicy(), svc.securityStatus(),
                svc.osVersion(), svc.allowedProtocols()
        );
    }

    private String getHeader(EnrichRequest req, String name) {
        return req.headers() != null ? req.headers().get(name) : null;
    }
}