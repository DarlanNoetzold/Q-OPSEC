package tech.noetzold.validation_send_api.model;

import jakarta.persistence.*;
import lombok.*;
import java.time.Instant;

@Entity
@Table(name = "encrypted_deliveries", indexes = {
        @Index(name = "idx_deliveries_request_id", columnList = "request_id", unique = true)
})
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class EncryptedDelivery {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name="request_id", length=120, nullable=false, unique=true)
    private String requestId;

    @Column(name="session_id", length=120, nullable=false)
    private String sessionId;

    @Column(name="selected_algorithm", length=80)
    private String selectedAlgorithm;

    @Column(name="crypto_algorithm", length=80, nullable=false)
    private String cryptoAlgorithm;

    @Column(name="crypto_nonce_b64", columnDefinition = "text", nullable=false)
    private String cryptoNonceB64;

    @Column(name="crypto_ciphertext_b64", columnDefinition = "text", nullable=false)
    private String cryptoCiphertextB64;

    @Column(name="crypto_expires_at")
    private Long cryptoExpiresAt;

    @Column(name="source_id", length=120)
    private String sourceId;

    @Column(name="origin_url", columnDefinition = "text", nullable=false)
    private String originUrl;

    @Column(name="status", length=40, nullable=false)
    private String status; // pending, delivered, failed

    @Column(name="error_message", columnDefinition = "text")
    private String errorMessage;

    @Column(name="created_at", nullable=false, updatable=false)
    private Instant createdAt;

    @Column(name="updated_at")
    private Instant updatedAt;

    @PrePersist
    public void prePersist() {
        Instant now = Instant.now();
        createdAt = now;
        updatedAt = now;
        if (status == null) status = "pending";
    }

    @PreUpdate
    public void preUpdate() {
        updatedAt = Instant.now();
    }
}