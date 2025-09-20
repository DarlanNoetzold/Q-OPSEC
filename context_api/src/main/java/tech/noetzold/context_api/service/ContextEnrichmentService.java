// src/main/java/tech/noetzold/context_api/service/ContextEnrichmentService.java
package tech.noetzold.context_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.ContextRecordRepository;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

@Slf4j
@Service
public class ContextEnrichmentService {

    private final SourceContextBuilder srcBuilder;
    private final DestinationContextBuilder destBuilder;
    private final RiskAssessmentFacade riskFacade;
    private final ConfidentialityFacade confFacade;
    private final ContextRecordRepository recordRepo;
    private final ObjectMapper objectMapper;

    public ContextEnrichmentService(SourceContextBuilder srcBuilder,
                                    DestinationContextBuilder destBuilder,
                                    RiskAssessmentFacade riskFacade,
                                    ConfidentialityFacade confFacade,
                                    ContextRecordRepository recordRepo,
                                    ObjectMapper objectMapper) {
        this.srcBuilder = srcBuilder;
        this.destBuilder = destBuilder;
        this.riskFacade = riskFacade;
        this.confFacade = confFacade;
        this.recordRepo = recordRepo;
        this.objectMapper = objectMapper;
    }

    @Transactional
    public EnrichResponse enrichAll(EnrichRequest req, HttpServletRequest httpReq) {
        log.info("EnrichAll called with requestId: {}", req.requestId());

        SourceContext source = srcBuilder.build(req.source_hint(), req, httpReq);
        DestinationContext dest = destBuilder.build(req.destination_hint(), req);

        RiskContext risk = riskFacade.getRisk(req.requestId(), source, dest);
        ContentConfidentiality conf = confFacade.classify(req.requestId(), req.content_pointer(), source, dest);

        try {
            var headersNode = objectMapper.valueToTree(req.headers() != null ? req.headers() : java.util.Map.of());
            var sourceNode = objectMapper.valueToTree(source);
            var destNode   = objectMapper.valueToTree(dest);
            var riskNode   = objectMapper.valueToTree(risk);
            var confNode   = objectMapper.valueToTree(conf);

            ContextRecord rec = new ContextRecord(
                    req.requestId(),
                    headersNode,
                    sourceNode,
                    destNode,
                    riskNode,
                    confNode
            );

            recordRepo.save(rec);
            log.info("ContextRecord saved with requestId: {}", req.requestId());
        } catch (Exception e) {
            log.warn("Error to persist ContextRecord", e);
        }

        return new EnrichResponse(req.requestId(), source, dest, risk, conf);
    }

    @Transactional
    public EnrichResponse enrichFromInterceptor(InterceptorPayload payload, HttpServletRequest httpReq) {
        log.info("EnrichFromInterceptor called with requestId: {}", payload.requestId());

        String reqId = (payload.requestId() != null && !payload.requestId().isBlank())
                ? payload.requestId()
                : "req_" + System.currentTimeMillis() + "_" + UUID.randomUUID().toString().substring(0, 8);

        Map<String, String> headers = new HashMap<>();
        if (payload.destinationId() != null) {
            headers.put("X-Service-Id", payload.destinationId());
        }

        if (payload.metadata() != null) {
            Object host = payload.metadata().get("host");
            Object path = payload.metadata().get("path");
            if (host instanceof String h && !h.isBlank()) {
                headers.put("X-Dest-Host", h);
            }
            if (path instanceof String p && !p.isBlank()) {
                headers.put("X-Request-Path", p);
            }
        }

        Map<String, Object> sourceHint = new HashMap<>();
        if (payload.sourceId() != null) {
            sourceHint.put("user_id", payload.sourceId());
        }
        if (payload.metadata() != null) {
            Object userAgent = payload.metadata().get("user_agent");
            Object deviceId = payload.metadata().get("device_id");
            if (userAgent instanceof String ua) {
                sourceHint.put("user_agent", ua);
            }
            if (deviceId instanceof String did) {
                sourceHint.put("device_id", did);
            }
        }

        Map<String, Object> destHint = new HashMap<>();
        if (payload.destinationId() != null) {
            destHint.put("service_id", payload.destinationId());
        }
        if (payload.metadata() != null) {
            Object host = payload.metadata().get("host");
            Object path = payload.metadata().get("path");
            if (host instanceof String h && !h.isBlank()) {
                destHint.put("host", h);
            }
            if (path instanceof String p && !p.isBlank()) {
                destHint.put("path", p);
            }
        }

        Map<String, Object> contentPointer = new HashMap<>();
        if (payload.content() != null && !payload.content().isBlank()) {
            contentPointer.put("sample_text", payload.content());
        }
        if (payload.metadata() != null && !payload.metadata().isEmpty()) {
            contentPointer.put("metadata", payload.metadata());
        }

        EnrichRequest enrichReq = new EnrichRequest(
                reqId,
                sourceHint.isEmpty() ? null : sourceHint,
                destHint.isEmpty() ? null : destHint,
                headers.isEmpty() ? null : headers,
                contentPointer.isEmpty() ? null : contentPointer
        );

        return enrichAll(enrichReq, httpReq);
    }
}