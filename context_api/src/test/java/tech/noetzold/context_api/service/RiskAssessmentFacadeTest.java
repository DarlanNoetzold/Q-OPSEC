package tech.noetzold.context_api.service;

import org.junit.jupiter.api.Test;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.RiskContext;
import tech.noetzold.context_api.model.SourceContext;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class RiskAssessmentFacadeTest {
    @Test
    void sendsDerivedSignalsAndReturnsRiskFromClient() {
        RiskServiceClient client = mock(RiskServiceClient.class);
        RiskContext expected = new RiskContext(0.8, "high", 0.6, Map.of(), 2, java.util.List.of(), "now", "v1");
        when(client.assessGeneral(eq("req-7"), anyMap())).thenAnswer(invocation -> {
            Map<String, Object> signals = invocation.getArgument(1);
            assertEquals("PrivateNet", signals.get("geo_region"));
            assertEquals("high", signals.get("exposure_level"));
            assertEquals("enabled", signals.get("mfa_status"));
            return Optional.of(expected);
        });
        SourceContext source = new SourceContext("192.168.1.3", "u", "d", "ua", "PrivateNet", "os", "laptop", "enabled", "healthy");
        DestinationContext destination = new DestinationContext("10.0.0.1", "svc", "api", "strict", "healthy", "os", java.util.List.of("TLS1.3"));

        RiskContext actual = new RiskAssessmentFacade(client).getRisk("req-7", source, destination);

        assertSame(expected, actual);
        verify(client).assessGeneral(eq("req-7"), anyMap());
    }

    @Test
    void returnsLowRiskFallbackWhenClientHasNoResult() {
        RiskServiceClient client = mock(RiskServiceClient.class);
        when(client.assessGeneral(anyString(), anyMap())).thenReturn(Optional.empty());

        RiskContext result = new RiskAssessmentFacade(client).getRisk("req-8", null, null);

        assertEquals(0.3, result.score());
        assertEquals("low", result.level());
        assertEquals("fallback", result.threat_intel().get("source"));
    }
}
