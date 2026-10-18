package tech.noetzold.validation_send_api;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.repository.EncryptedDeliveryRepository;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest(properties = {
        "spring.datasource.url=jdbc:h2:mem:send_api_full;DB_CLOSE_DELAY=-1",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "spring.jpa.database-platform=org.hibernate.dialect.H2Dialect",
        "spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.H2Dialect"
})
@AutoConfigureMockMvc
class ValidationSendApiIntegrationTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private EncryptedDeliveryRepository repository;

    @BeforeEach
    void cleanDatabase() {
        repository.deleteAll();
    }

    @Test
    void persistsDeliveryAndMarksItFailedWhenOriginCannotBeReached() throws Exception {
        mockMvc.perform(post("/validation/send")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"requestId\":\"integration-1\",\"sessionId\":\"session-1\","
                                + "\"selectedAlgorithm\":\"X25519\",\"cryptoNonceB64\":\"nonce\","
                                + "\"cryptoCiphertextB64\":\"ciphertext\",\"cryptoAlgorithm\":\"AES-GCM\","
                                + "\"cryptoExpiresAt\":123456,\"sourceId\":\"source-1\","
                                + "\"originUrl\":\"http://127.0.0.1:1/unreachable\"}"))
                .andExpect(status().isBadGateway())
                .andExpect(content().json("{\"status\":\"failed\"}"));

        EncryptedDelivery delivery = repository.findByRequestId("integration-1").orElseThrow();
        assertThat(delivery.getSessionId()).isEqualTo("session-1");
        assertThat(delivery.getCryptoAlgorithm()).isEqualTo("AES-GCM");
        assertThat(delivery.getStatus()).isEqualTo("failed");
        assertThat(delivery.getErrorMessage()).startsWith("Failed to forward to origin:");
        assertThat(delivery.getCreatedAt()).isNotNull();
        assertThat(delivery.getUpdatedAt()).isNotNull();
    }
}
