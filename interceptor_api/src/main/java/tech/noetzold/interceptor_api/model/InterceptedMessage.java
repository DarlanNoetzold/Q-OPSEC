package tech.noetzold.interceptor_api.model;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "intercepted_messages")
@Getter @Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterceptedMessage {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "request_id", length = 120, nullable = false, unique = true)
    private String requestId;

    @Column(name = "message", nullable = false, columnDefinition = "text")
    private String message;

    @Column(name = "source_id", length = 120)
    private String sourceId;

    @Column(name = "destination_id", length = 120)
    private String destinationId;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "metadata_json", columnDefinition = "jsonb")
    private JsonNode metadataJson;

    @Column(name = "status", length = 40)
    private String status; // e.g. "intercepted", "processed"

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "updated_at")
    private Instant updatedAt;

    @PrePersist
    public void prePersist() {
        if (createdAt == null) createdAt = Instant.now();
        updatedAt = createdAt;
    }

    @PreUpdate
    public void preUpdate() {
        updatedAt = Instant.now();
    }
}