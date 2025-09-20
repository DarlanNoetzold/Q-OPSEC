// src/main/java/tech/noetzold/context_api/persistence/ContextRecord.java
package tech.noetzold.context_api.model;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.persistence.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.Instant;

@Entity
@Table(name = "context_records")
public class ContextRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "requestId", length = 120)
    private String requestId;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "headers_json", columnDefinition = "jsonb")
    private JsonNode headersJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "source_json", nullable = false, columnDefinition = "jsonb")
    private JsonNode sourceJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "destination_json", nullable = false, columnDefinition = "jsonb")
    private JsonNode destinationJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "risk_json", nullable = false, columnDefinition = "jsonb")
    private JsonNode riskJson;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "confidentiality_json", nullable = false, columnDefinition = "jsonb")
    private JsonNode confidentialityJson;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @PrePersist
    public void prePersist() {
        if (createdAt == null) createdAt = Instant.now();
    }

    public ContextRecord() {}

    public ContextRecord(String requestId,
                         JsonNode headersJson,
                         JsonNode sourceJson,
                         JsonNode destinationJson,
                         JsonNode riskJson,
                         JsonNode confidentialityJson) {
        this.requestId = requestId;
        this.headersJson = headersJson;
        this.sourceJson = sourceJson;
        this.destinationJson = destinationJson;
        this.riskJson = riskJson;
        this.confidentialityJson = confidentialityJson;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getRequestId() {
        return requestId;
    }

    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    public JsonNode getHeadersJson() {
        return headersJson;
    }

    public void setHeadersJson(JsonNode headersJson) {
        this.headersJson = headersJson;
    }

    public JsonNode getSourceJson() {
        return sourceJson;
    }

    public void setSourceJson(JsonNode sourceJson) {
        this.sourceJson = sourceJson;
    }

    public JsonNode getDestinationJson() {
        return destinationJson;
    }

    public void setDestinationJson(JsonNode destinationJson) {
        this.destinationJson = destinationJson;
    }

    public JsonNode getRiskJson() {
        return riskJson;
    }

    public void setRiskJson(JsonNode riskJson) {
        this.riskJson = riskJson;
    }

    public JsonNode getConfidentialityJson() {
        return confidentialityJson;
    }

    public void setConfidentialityJson(JsonNode confidentialityJson) {
        this.confidentialityJson = confidentialityJson;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }
}