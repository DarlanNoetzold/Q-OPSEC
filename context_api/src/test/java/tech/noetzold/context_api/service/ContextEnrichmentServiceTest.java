package tech.noetzold.context_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.ContextRecordRepository;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ContextEnrichmentServiceTest {
    @Mock SourceContextBuilder sourceBuilder;
    @Mock DestinationContextBuilder destinationBuilder;
    @Mock RiskAssessmentFacade riskFacade;
    @Mock ConfidentialityFacade confidentialityFacade;
    @Mock ContextRecordRepository recordRepository;
    @Mock HttpServletRequest httpRequest;

    private ContextEnrichmentService service() {
        return new ContextEnrichmentService(sourceBuilder, destinationBuilder, riskFacade,
                confidentialityFacade, recordRepository, new ObjectMapper());
    }

    @Test
    void enrichFromInterceptorMapsPayloadIntoContextAndPersistsResult() {
        SourceContext source = new SourceContext("192.168.1.4", "user-1", "device-1", "agent", "PrivateNet", "os", "laptop", "enabled", "healthy");
        DestinationContext destination = new DestinationContext("10.0.0.2", "service-1", "api", "strict", "healthy", "os", List.of("TLS1.3"));
        RiskContext risk = new RiskContext(0.2, "low", 0.1, Map.of(), 0, List.of(), "now", "risk-v1");
        ContentConfidentiality confidentiality = new ContentConfidentiality("internal", 0.4, List.of(), List.of(), List.of(), null, null, "conf-v1");
        when(sourceBuilder.build(any(), any(), eq(httpRequest))).thenReturn(source);
        when(destinationBuilder.build(any(), any())).thenReturn(destination);
        when(riskFacade.getRisk(eq("req-interceptor"), eq(source), eq(destination))).thenReturn(risk);
        when(confidentialityFacade.classify(eq("req-interceptor"), anyMap(), eq(source), eq(destination))).thenReturn(confidentiality);
        InterceptorPayload payload = new InterceptorPayload("req-interceptor", "user-1", "service-1", "sample payload",
                Map.of("host", "api.example.test", "path", "/v1", "user_agent", "agent", "device_id", "device-1"));

        EnrichResponse response = service().enrichFromInterceptor(payload, httpRequest);

        assertEquals("req-interceptor", response.request_id());
        assertSame(source, response.source());
        ArgumentCaptor<EnrichRequest> requestCaptor = ArgumentCaptor.forClass(EnrichRequest.class);
        verify(sourceBuilder).build(any(), requestCaptor.capture(), eq(httpRequest));
        EnrichRequest mapped = requestCaptor.getValue();
        assertEquals("user-1", mapped.source_hint().get("user_id"));
        assertEquals("device-1", mapped.source_hint().get("device_id"));
        assertEquals("service-1", mapped.destination_hint().get("service_id"));
        assertEquals("api.example.test", mapped.destination_hint().get("host"));
        assertEquals("/v1", mapped.destination_hint().get("path"));
        assertEquals("sample payload", mapped.content_pointer().get("sample_text"));
        assertEquals("api.example.test", mapped.headers().get("X-Dest-Host"));
        verify(riskFacade).getRisk("req-interceptor", source, destination);
        verify(recordRepository).save(any(ContextRecord.class));
    }

    @Test
    void enrichAllUsesProvidedRequestIdAndReturnsCombinedContext() {
        EnrichRequest request = new EnrichRequest("req-direct", null, null, Map.of("X-Trace", "trace"), null);
        SourceContext source = new SourceContext("0.0.0.0", null, null, null, null, "unknown", "unknown", "unknown", "unknown");
        DestinationContext destination = new DestinationContext("0.0.0.0", null, "unknown", "medium", "unknown", "unknown", List.of());
        RiskContext risk = new RiskContext(0.1, "low", 0.0, Map.of(), 0, List.of(), "now", "risk-v1");
        ContentConfidentiality confidentiality = new ContentConfidentiality("internal", 0.4, List.of(), List.of(), List.of(), null, null, "conf-v1");
        when(sourceBuilder.build(null, request, httpRequest)).thenReturn(source);
        when(destinationBuilder.build(null, request)).thenReturn(destination);
        when(riskFacade.getRisk("req-direct", source, destination)).thenReturn(risk);
        when(confidentialityFacade.classify("req-direct", null, source, destination)).thenReturn(confidentiality);

        EnrichResponse response = service().enrichAll(request, httpRequest);

        assertEquals("req-direct", response.request_id());
        assertSame(risk, response.risk());
        ArgumentCaptor<ContextRecord> savedCaptor = ArgumentCaptor.forClass(ContextRecord.class);
        verify(recordRepository).save(savedCaptor.capture());
        assertEquals("req-direct", savedCaptor.getValue().getRequestId());
        assertEquals("trace", savedCaptor.getValue().getHeadersJson().get("X-Trace").asText());
    }
}
