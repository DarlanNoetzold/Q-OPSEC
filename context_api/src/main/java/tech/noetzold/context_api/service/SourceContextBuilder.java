package tech.noetzold.context_api.service;

import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.DeviceRepository;

import java.util.Map;
import java.util.Optional;

@Component
public class SourceContextBuilder {

    private final GeoIpService geoIpService;
    private final DeviceRepository deviceRepo;

    public SourceContextBuilder(GeoIpService geoIpService, DeviceRepository deviceRepo) {
        this.geoIpService = geoIpService;
        this.deviceRepo = deviceRepo;
    }

    public SourceContext build(Map<String,Object> sourceHint, EnrichRequest req, ServerHttpRequest httpReq) {
        var sh = Optional.ofNullable(sourceHint).orElse(Map.of());
        String srcIp = (String) sh.getOrDefault("ip",
                httpReq.getRemoteAddress() != null ?
                        httpReq.getRemoteAddress().getAddress().getHostAddress() : "0.0.0.0");
        String userAgent = (String) sh.getOrDefault("user_agent", getHeader(req, "User-Agent"));
        String userId = (String) sh.getOrDefault("user_id", getHeader(req, "X-User-Id"));
        String deviceId = (String) sh.getOrDefault("device_id", getHeader(req, "X-Device-Id"));

        var geo = geoIpService.lookupCountry(srcIp).orElse(null);
        var devMeta = deviceRepo.findByDeviceId(deviceId)
                .orElse(new DeviceMeta("unknown","unknown","unknown","unknown"));

        return new SourceContext(
                srcIp, userId, deviceId, userAgent, geo,
                devMeta.osVersion(), devMeta.deviceType(),
                devMeta.mfaStatus(), devMeta.securityStatus()
        );
    }

    private String getHeader(EnrichRequest req, String name) {
        return req.headers() != null ? req.headers().get(name) : null;
    }
}