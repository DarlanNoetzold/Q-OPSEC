package tech.noetzold.context_api;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.RiskContext;
import tech.noetzold.context_api.service.ConfidentialityServiceClient;
import tech.noetzold.context_api.service.RiskServiceClient;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@Transactional
class ContextEnrichmentIntegrationTest {
    @Autowired MockMvc mockMvc;
    @MockitoBean RiskServiceClient riskServiceClient;
    @MockitoBean ConfidentialityServiceClient confidentialityServiceClient;

    @Test
    void enrichEndpointBuildsContextPersistsRecordAndMakesItRetrievable() throws Exception {
        when(riskServiceClient.assessGeneral(anyString(), anyMap())).thenReturn(Optional.of(
                new RiskContext(0.2, "low", 0.1, Map.of(), 0, List.of(), "now", "risk-test")));
        when(confidentialityServiceClient.classify(anyString(), anyMap(), any(), any())).thenReturn(Optional.of(
                new ContentConfidentiality("internal", 0.4, List.of(), List.of(), List.of(), null, null, "conf-test")));

        mockMvc.perform(post("/context/enrich")
                        .contentType("application/json")
                        .content("""
                                {
                                  "requestId":"integration-context-001",
                                  "source_hint":{"ip":"192.168.10.20","user_id":"user-42"},
                                  "destination_hint":{"service_id":"service-7","host":"api.example.test","path":"/v1/data"},
                                  "headers":{"X-Trace-Id":"trace-123"},
                                  "content_pointer":{"sample_text":"internal test data"}
                                }
                                """))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.request_id").value("integration-context-001"))
                .andExpect(jsonPath("$.source.ip").value("192.168.10.20"))
                .andExpect(jsonPath("$.source.user_id").value("user-42"))
                .andExpect(jsonPath("$.source.geo").value("PrivateNet"))
                .andExpect(jsonPath("$.destination.service_id").value("service-7"))
                .andExpect(jsonPath("$.risk.level").value("low"))
                .andExpect(jsonPath("$.confidentiality.classification").value("internal"));

        mockMvc.perform(get("/context/record").param("requestId", "integration-context-001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.requestId").value("integration-context-001"))
                .andExpect(jsonPath("$.headersJson.X-Trace-Id").value("trace-123"))
                .andExpect(jsonPath("$.sourceJson.ip").value("192.168.10.20"))
                .andExpect(jsonPath("$.destinationJson.service_id").value("service-7"))
                .andExpect(jsonPath("$.riskJson.level").value("low"))
                .andExpect(jsonPath("$.confidentialityJson.classification").value("internal"));
    }
}
