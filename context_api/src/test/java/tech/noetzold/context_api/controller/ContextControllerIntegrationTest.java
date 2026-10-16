package tech.noetzold.context_api.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.ContextRecordRepository;
import tech.noetzold.context_api.service.ContextEnrichmentService;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(ContextController.class)
class ContextControllerIntegrationTest {
    @Autowired MockMvc mockMvc;
    @MockitoBean ContextEnrichmentService service;
    @MockitoBean ContextRecordRepository recordRepository;

    @Test
    void enrichEndpointAcceptsRequestAndReturnsCombinedContext() throws Exception {
        SourceContext source = new SourceContext("192.168.1.2", "user-1", null, "agent", "PrivateNet", "unknown", "unknown", "unknown", "unknown");
        DestinationContext destination = new DestinationContext("10.0.0.1", "svc-1", "api", "medium", "healthy", "Linux", List.of("TLS1.3"));
        RiskContext risk = new RiskContext(0.2, "low", 0.1, Map.of(), 0, List.of(), "now", "risk-v1");
        ContentConfidentiality confidentiality = new ContentConfidentiality("internal", 0.4, List.of(), List.of(), List.of(), null, null, "conf-v1");
        when(service.enrichAll(any(EnrichRequest.class), any())).thenReturn(
                new EnrichResponse("req-http", source, destination, risk, confidentiality));

        mockMvc.perform(post("/context/enrich")
                        .contentType("application/json")
                        .content("""
                                {"requestId":"req-http","source_hint":{"user_id":"user-1"},"destination_hint":{"service_id":"svc-1"}}
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.request_id").value("req-http"))
                .andExpect(jsonPath("$.source.ip").value("192.168.1.2"))
                .andExpect(jsonPath("$.destination.service_id").value("svc-1"))
                .andExpect(jsonPath("$.risk.level").value("low"))
                .andExpect(jsonPath("$.confidentiality.classification").value("internal"));
    }

    @Test
    void recordEndpointReturnsNotFoundWhenRequestIdDoesNotExist() throws Exception {
        when(recordRepository.findByRequestId("missing")).thenReturn(Optional.empty());

        mockMvc.perform(get("/context/record").param("requestId", "missing"))
                .andExpect(status().isNotFound());
    }
}
