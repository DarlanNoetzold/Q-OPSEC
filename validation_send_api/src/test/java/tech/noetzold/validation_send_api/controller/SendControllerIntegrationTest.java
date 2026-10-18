package tech.noetzold.validation_send_api.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.service.ForwardService;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(properties = {
        "spring.datasource.url=jdbc:h2:mem:send_api_http;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "spring.jpa.database-platform=org.hibernate.dialect.H2Dialect",
        "spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.H2Dialect"
})
@AutoConfigureMockMvc
class SendControllerIntegrationTest {
    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private ForwardService forwardService;

    @Test
    void forwardsValidPayloadAndReturnsDelivered() throws Exception {
        EncryptedDelivery delivery = EncryptedDelivery.builder().requestId("request-1").build();
        when(forwardService.receiveAndPersist(any())).thenReturn(delivery);

        mockMvc.perform(post("/validation/send")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(validPayload()))
                .andExpect(status().isOk())
                .andExpect(content().json("{\"status\":\"delivered\"}"));

        verify(forwardService).forwardToOriginOrThrow(any());
        verify(forwardService).markDelivered(delivery);
        verify(forwardService, never()).markFailed(any(), any());
    }

    @Test
    void returnsBadRequestAndDoesNotCallServiceWhenRequiredFieldIsBlank() throws Exception {
        String payload = validPayload().replace("\"requestId\":\"request-1\"", "\"requestId\":\" \"");

        mockMvc.perform(post("/validation/send")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(payload))
                .andExpect(status().isBadRequest());

        verifyNoInteractions(forwardService);
    }

    @Test
    void recordsFailureAndReturnsBadGatewayWhenOriginForwardingFails() throws Exception {
        EncryptedDelivery delivery = EncryptedDelivery.builder().requestId("request-1").build();
        when(forwardService.receiveAndPersist(any())).thenReturn(delivery);
        doThrow(new IllegalStateException("origin unavailable"))
                .when(forwardService).forwardToOriginOrThrow(any());

        mockMvc.perform(post("/validation/send")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(validPayload()))
                .andExpect(status().isBadGateway())
                .andExpect(content().json("{\"status\":\"failed\",\"error\":\"origin unavailable\"}"));

        verify(forwardService).markFailed(delivery, "origin unavailable");
        verify(forwardService, never()).markDelivered(any());
    }

    private static String validPayload() {
        return "{\"requestId\":\"request-1\",\"sessionId\":\"session-1\","
                + "\"selectedAlgorithm\":\"X25519\",\"cryptoNonceB64\":\"nonce\","
                + "\"cryptoCiphertextB64\":\"ciphertext\",\"cryptoAlgorithm\":\"AES-GCM\","
                + "\"cryptoExpiresAt\":123456,\"sourceId\":\"source-1\","
                + "\"originUrl\":\"http://localhost:8080/origin\"}";
    }
}
