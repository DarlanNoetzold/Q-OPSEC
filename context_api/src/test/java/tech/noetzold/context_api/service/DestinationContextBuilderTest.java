package tech.noetzold.context_api.service;

import org.junit.jupiter.api.Test;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.EnrichRequest;
import tech.noetzold.context_api.model.ServiceMeta;
import tech.noetzold.context_api.repository.ServiceRepository;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

class DestinationContextBuilderTest {
    @Test
    void usesDestinationHintsAndResolvedServiceMetadata() {
        ServiceRepository repository = mock(ServiceRepository.class);
        when(repository.findByServiceIdAndHost("svc-1", "api.example.test", "/v1/data"))
                .thenReturn(Optional.of(new ServiceMeta("10.1.2.3", "api", "strict", "healthy", "Linux", List.of("TLS1.3"))));

        DestinationContext result = new DestinationContextBuilder(repository).build(
                Map.of("service_id", "svc-1", "host", "api.example.test", "path", "/v1/data"),
                new EnrichRequest("req-1", null, null, null, null));

        assertEquals("10.1.2.3", result.ip());
        assertEquals("svc-1", result.service_id());
        assertEquals("api", result.service_type());
        assertEquals("strict", result.security_policy());
        assertEquals(List.of("TLS1.3"), result.allowed_protocols());
        verify(repository).findByServiceIdAndHost("svc-1", "api.example.test", "/v1/data");
    }

    @Test
    void usesHeadersAndSafeDefaultsWhenServiceIsUnknown() {
        ServiceRepository repository = mock(ServiceRepository.class);
        when(repository.findByServiceIdAndHost("svc-unknown", "unknown.test", "/"))
                .thenReturn(Optional.empty());
        EnrichRequest request = new EnrichRequest("req-2", null, null,
                Map.of("X-Service-Id", "svc-unknown", "X-Dest-Host", "unknown.test", "X-Request-Path", "/"), null);

        DestinationContext result = new DestinationContextBuilder(repository).build(null, request);

        assertEquals("0.0.0.0", result.ip());
        assertEquals("svc-unknown", result.service_id());
        assertEquals("unknown", result.service_type());
        assertEquals("medium", result.security_policy());
        assertEquals(List.of("TLS1.2", "TLS1.3"), result.allowed_protocols());
    }
}
