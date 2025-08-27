// src/main/java/tech/noetzold/context_api/persistence/ContextRecord.java
package tech.noetzold.context_api.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "context_records")
public class ContextRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "request_id", length = 120)
    private String requestId;

    @Lob
    @Column(name = "headers_json")
    private String headersJson;

    @Lob
    @Column(name = "source_json", nullable = false)
    private String sourceJson;

    @Lob
    @Column(name = "destination_json", nullable = false)
    private String destinationJson;

    @Lob
    @Column(name = "risk_json", nullable = false)
    private String riskJson;

    @Lob
    @Column(name = "confidentiality_json", nullable = false)
    private String confidentialityJson;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt = Instant.now();

    public ContextRecord() {}

    public ContextRecord(String requestId, String headersJson, String sourceJson, String destinationJson,
                         String riskJson, String confidentialityJson) {
        this.requestId = requestId;
        this.headersJson = headersJson;
        this.sourceJson = sourceJson;
        this.destinationJson = destinationJson;
        this.riskJson = riskJson;
        this.confidentialityJson = confidentialityJson;
    }

    // getters e setters
    public Long getId() { return id; }
    public String getRequestId() { return requestId; }
    public void setRequestId(String requestId) { this.requestId = requestId; }
    public String getHeadersJson() { return headersJson; }
    public void setHeadersJson(String headersJson) { this.headersJson = headersJson; }
    public String getSourceJson() { return sourceJson; }
    public void setSourceJson(String sourceJson) { this.sourceJson = sourceJson; }
    public String getDestinationJson() { return destinationJson; }
    public void setDestinationJson(String destinationJson) { this.destinationJson = destinationJson; }
    public String getRiskJson() { return riskJson; }
    public void setRiskJson(String riskJson) { this.riskJson = riskJson; }
    public String getConfidentialityJson() { return confidentialityJson; }
    public void setConfidentialityJson(String confidentialityJson) { this.confidentialityJson = confidentialityJson; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}