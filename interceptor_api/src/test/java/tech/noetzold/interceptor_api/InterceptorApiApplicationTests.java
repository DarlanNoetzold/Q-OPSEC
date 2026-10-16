package tech.noetzold.interceptor_api;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;
import tech.noetzold.interceptor_api.model.ContextResponse;
import tech.noetzold.interceptor_api.repository.InterceptedMessageRepository;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest(properties = {
        "spring.datasource.url=jdbc:h2:mem:interceptor-test;DB_CLOSE_DELAY=-1;MODE=PostgreSQL",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "spring.jpa.database-platform=org.hibernate.dialect.H2Dialect",
        "classification.forward.delay.ms=0",
        "classification.api.url=http://127.0.0.1:1/predict"
})
@AutoConfigureMockMvc
class InterceptorApiApplicationTests {
    @Autowired MockMvc mockMvc;
    @Autowired ObjectMapper objectMapper;
    @Autowired InterceptedMessageRepository messageRepository;
    @MockBean RestTemplate restTemplate;

    @BeforeEach
    void setUp() {
        messageRepository.deleteAll();
        when(restTemplate.postForObject(any(String.class), any(), org.mockito.ArgumentMatchers.eq(ContextResponse.class)))
                .thenReturn(ContextResponse.builder()
                        .requestId("context-request")
                        .risk(Map.of("level", "low"))
                        .confidentiality(Map.of("level", "public"))
                        .build());
        when(restTemplate.postForEntity(any(String.class), any(), org.mockito.ArgumentMatchers.eq(String.class)))
                .thenReturn(org.springframework.http.ResponseEntity.ok("{}"));
    }

    @Test
    void healthEndpointReturnsOk() throws Exception {
        mockMvc.perform(get("/health"))
                .andExpect(status().isOk())
                .andExpect(content().json("{\"status\":\"ok\"}"));
    }

    @Test
    void interceptPersistsMessageAndExposesMessageAndBase64Payload() throws Exception {
        String request = objectMapper.writeValueAsString(Map.of(
                "message", "hello integration",
                "sourceId", "source-1",
                "destinationId", "destination-1",
                "metadata", Map.of("channel", "integration-test")
        ));

        String response = mockMvc.perform(post("/intercept")
                        .contentType(MediaType.APPLICATION_JSON)
                        .header("X-Source-Id", "source-1")
                        .header("X-Destination-Id", "destination-1")
                        .content(request))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("OK"))
                .andExpect(jsonPath("$.requestId").isNotEmpty())
                .andExpect(jsonPath("$.context.risk.level").value("low"))
                .andReturn().getResponse().getContentAsString();

        String requestId = objectMapper.readTree(response).get("requestId").asText();
        mockMvc.perform(get("/intercept/message").param("request_id", requestId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.request_id").value(requestId))
                .andExpect(jsonPath("$.message").value("hello integration"))
                .andExpect(jsonPath("$.source_id").value("source-1"))
                .andExpect(jsonPath("$.destination_id").value("destination-1"))
                .andExpect(jsonPath("$.status").value("processed"))
                .andExpect(jsonPath("$.metadata").value("{\"channel\":\"integration-test\"}"));

        mockMvc.perform(get("/intercept/payload").param("request_id", requestId))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.request_id").value(requestId))
                .andExpect(jsonPath("$.payload_b64").value("aGVsbG8gaW50ZWdyYXRpb24="))
                .andExpect(jsonPath("$.content_type").value("application/json"));
    }

    @Test
    void messageAndPayloadEndpointsReturnNotFoundForUnknownRequest() throws Exception {
        mockMvc.perform(get("/intercept/message").param("request_id", "missing"))
                .andExpect(status().isNotFound());
        mockMvc.perform(get("/intercept/payload").param("request_id", "missing"))
                .andExpect(status().isNotFound());
    }
}
