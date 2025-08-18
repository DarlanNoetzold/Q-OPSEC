package tech.noetzold.context_api.service;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.DeviceMeta;
import tech.noetzold.context_api.model.EnrichRequest;
import tech.noetzold.context_api.model.SourceContext;
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

    public SourceContext build(Map<String, Object> sourceHint, EnrichRequest req, HttpServletRequest httpReq) {
        var sh = Optional.ofNullable(sourceHint).orElse(Map.of());

        String srcIp = (String) sh.get("ip");
        if (srcIp == null || srcIp.isBlank()) {
            srcIp = resolveClientIp(httpReq);
        }
        if (srcIp == null || srcIp.isBlank()) {
            srcIp = "0.0.0.0";
        }

        // Headers via servlet request, com hint como override
        String userAgent = (String) sh.getOrDefault("user_agent", getHeader(httpReq, req, "User-Agent"));
        String userId = (String) sh.getOrDefault("user_id", getHeader(httpReq, req, "X-User-Id"));
        String deviceId = (String) sh.getOrDefault("device_id", getHeader(httpReq, req, "X-Device-Id"));

        var geo = geoIpService.lookupCountry(srcIp).orElse(null);
        var devMeta = (deviceId != null && !deviceId.isBlank())
                ? deviceRepo.findByDeviceId(deviceId)
                .orElse(new DeviceMeta("unknown","unknown","unknown","unknown"))
                : new DeviceMeta("unknown","unknown","unknown","unknown");

        return new SourceContext(
                srcIp, userId, deviceId, userAgent, geo,
                devMeta.osVersion(), devMeta.deviceType(),
                devMeta.mfaStatus(), devMeta.securityStatus()
        );
    }

    private String resolveClientIp(HttpServletRequest request) {
        String xff = request.getHeader("X-Forwarded-For");
        if (xff != null && !xff.isBlank()) {
            String first = xff.split(",")[0].trim();
            if (!first.isBlank()) return first;
        }
        String xri = request.getHeader("X-Real-IP");
        if (xri != null && !xri.isBlank()) return xri;
        return request.getRemoteAddr();
    }

    private String getHeader(HttpServletRequest httpReq, EnrichRequest req, String name) {
        String v = httpReq != null ? httpReq.getHeader(name) : null;
        if (v != null) return v;
        return req.headers() != null ? req.headers().get(name) : null;
    }
}