package tech.noetzold.context_api.service;

import org.junit.jupiter.api.Test;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ConfidentialityFacadeTest {
    @Test
    void forwardsContentAndContextToClassificationClient() {
        ConfidentialityServiceClient client = mock(ConfidentialityServiceClient.class);
        SourceContext source = new SourceContext("192.168.1.3", "u", "d", "ua", "PrivateNet", "os", "laptop", "enabled", "healthy");
        DestinationContext destination = new DestinationContext("10.0.0.1", "svc", "api", "strict", "healthy", "os", java.util.List.of("TLS1.3"));
        Map<String, Object> content = Map.of("sample_text", "hello");
        ContentConfidentiality expected = new ContentConfidentiality("public", 0.1, java.util.List.of(), java.util.List.of(), java.util.List.of(), null, null, "v1");
        when(client.classify("req-9", content, source, destination)).thenReturn(Optional.of(expected));

        ContentConfidentiality actual = new ConfidentialityFacade(client).classify("req-9", content, source, destination);

        assertSame(expected, actual);
        verify(client).classify("req-9", content, source, destination);
    }

    @Test
    void usesInternalFallbackWhenClientHasNoResult() {
        ConfidentialityServiceClient client = mock(ConfidentialityServiceClient.class);
        when(client.classify(eq("req-10"), anyMap(), isNull(), isNull())).thenReturn(Optional.empty());

        ContentConfidentiality result = new ConfidentialityFacade(client).classify("req-10", null, null, null);

        assertEquals("internal", result.classification());
        assertEquals(0.4, result.score());
        verify(client).classify(eq("req-10"), argThat(Map::isEmpty), isNull(), isNull());
    }
}
