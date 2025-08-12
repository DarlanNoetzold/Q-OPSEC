package tech.noetzold.context_api.service;

import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Service;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.DeviceRepository;
import tech.noetzold.context_api.repository.ServiceRepository;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class ContextEnrichmentService {

    private final RiskServiceClient riskClient;
    private final ConfidentialityServiceClient confClient;
    private final GeoIpService geoIpService;
    private final DeviceRepository deviceRepo;
    private final ServiceRepository serviceRepo;

    public ContextEnrichmentService(RiskServiceClient riskClient, ConfidentialityServiceClient confClient,
                                    GeoIpService geoIpService, DeviceRepository deviceRepo, ServiceRepository serviceRepo) {
        this.riskClient = riskClient;
        this.confClient = confClient;
        this.geoIpService = geoIpService;
        this.deviceRepo = deviceRepo;
        this.serviceRepo = serviceRepo;
    }

    public EnrichResponse enrichAll(EnrichRequest req, ServerHttpRequest httpReq) {
        // Source
        var sh = Optional.ofNullable(req.source_hint()).orElse(Map.of());
        String srcIp = (String) sh.getOrDefault("ip",
                httpReq.getRemoteAddress() != null ? httpReq.getRemoteAddress().getAddress().getHostAddress() : "0.0.0.0");
        String userAgent = (String) sh.getOrDefault("user_agent", header(req, "User-Agent"));
        String userId = (String) sh.getOrDefault("user_id", header(req, "X-User-Id"));
        String deviceId = (String) sh.getOrDefault("device_id", header(req, "X-Device-Id"));

        var geo = geoIpService.lookupCountry(srcIp).orElse(null);
        var devMeta = deviceRepo.findByDeviceId(deviceId).orElse(new DeviceMeta("unknown","unknown","unknown","unknown"));

        var source = new SourceContext(
                srcIp, userId, deviceId, userAgent, geo,
                devMeta.osVersion(), devMeta.deviceType(), devMeta.mfaStatus(), devMeta.securityStatus()
        );

        // Destination
        var dh = Optional.ofNullable(req.destination_hint()).orElse(Map.of());
        String destHost = (String) dh.getOrDefault("host", header(req, "X-Dest-Host"));
        String serviceId = (String) dh.getOrDefault("service_id", header(req, "X-Service-Id"));
        String path = (String) dh.getOrDefault("path", header(req, "X-Request-Path"));

        var svc = serviceRepo.findByServiceIdAndHost(serviceId, destHost, path)
                .orElse(new ServiceMeta("0.0.0.0","unknown","medium","unknown","unknown", List.of("TLS1.2","TLS1.3")));

        var destination = new DestinationContext(
                svc.ip(), serviceId, svc.serviceType(), svc.securityPolicy(),
                svc.securityStatus(), svc.osVersion(), svc.allowedProtocols()
        );

        // Risk + Confidentiality
        var risk = riskClient.assess(req.request_id(), source, destination)
                .orElse(new RiskContext(0.3, "low", 0.1, Map.of("source","fallback"), 0, List.of(), Instant.now().toString(), "risk-fb-0.0.1"));

        var conf = confClient.classify(req.request_id(), req.content_pointer(), source, destination)
                .orElse(new ContentConfidentiality("internal", 0.4, List.of(), List.of(), List.of(), null, null, "conf-fb-0.0.1"));

        return new EnrichResponse(req.request_id(), source, destination, risk, conf);
    }

    private String header(EnrichRequest req, String name) {
        return req.headers() != null ? req.headers().get(name) : null;
    }
}