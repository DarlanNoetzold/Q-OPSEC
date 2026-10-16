package tech.noetzold.context_api.service;

import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.noetzold.context_api.model.DeviceMeta;
import tech.noetzold.context_api.model.EnrichRequest;
import tech.noetzold.context_api.model.SourceContext;
import tech.noetzold.context_api.repository.DeviceRepository;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SourceContextBuilderTest {
    @Mock DeviceRepository deviceRepository;
    @Mock HttpServletRequest httpRequest;

    @Test
    void resolvesForwardedIpAndBuildsSourceFromHeadersAndDeviceMetadata() {
        when(httpRequest.getHeader("X-Forwarded-For")).thenReturn("192.168.4.12, 10.0.0.1");
        when(httpRequest.getHeader("User-Agent")).thenReturn("agent/1");
        when(httpRequest.getHeader("X-User-Id")).thenReturn("user-7");
        when(httpRequest.getHeader("X-Device-Id")).thenReturn("device-9");
        when(deviceRepository.findByDeviceId("device-9"))
                .thenReturn(Optional.of(new DeviceMeta("Linux 6", "laptop", "enabled", "healthy")));

        SourceContext result = new SourceContextBuilder(new GeoIpService(), deviceRepository)
                .build(null, new EnrichRequest("req-1", null, null, null, null), httpRequest);

        assertEquals("192.168.4.12", result.ip());
        assertEquals("PrivateNet", result.geo());
        assertEquals("user-7", result.user_id());
        assertEquals("device-9", result.device_id());
        assertEquals("agent/1", result.user_agent());
        assertEquals("Linux 6", result.os_version());
        assertEquals("laptop", result.device_type());
        assertEquals("enabled", result.mfa_status());
        assertEquals("healthy", result.security_status());
        verify(deviceRepository).findByDeviceId("device-9");
    }

    @Test
    void requestHintsOverrideHeadersAndMissingIpFallsBackToZeroAddress() {
        Map<String, Object> sourceHint = Map.of("user_id", "hint-user", "ip", "203.0.113.4");
        EnrichRequest req = new EnrichRequest("req-2", sourceHint, null,
                Map.of("X-User-Id", "header-user"), null);

        SourceContext result = new SourceContextBuilder(new GeoIpService(), deviceRepository)
                .build(sourceHint, req, httpRequest);

        assertEquals("203.0.113.4", result.ip());
        assertEquals("hint-user", result.user_id());
        assertEquals("Unknown", result.geo());
        assertEquals("unknown", result.device_type());
        verifyNoInteractions(deviceRepository);
    }
}
