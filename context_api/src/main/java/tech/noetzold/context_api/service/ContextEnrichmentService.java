// src/main/java/tech/noetzold/context_api/service/ContextEnrichmentService.java
package tech.noetzold.context_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.repository.ContextRecordRepository;

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
        SourceContext source = srcBuilder.build(req.source_hint(), req, httpReq);
        DestinationContext dest = destBuilder.build(req.destination_hint(), req);

        RiskContext risk = riskFacade.getRisk(req.request_id(), source, dest);
        ContentConfidentiality conf = confFacade.classify(req.request_id(), req.content_pointer(), source, dest);

        // Persistência
        try {
            String headersJson = req.headers() != null ? objectMapper.writeValueAsString(req.headers()) : "{}";
            String sourceJson = objectMapper.writeValueAsString(source);
            String destJson = objectMapper.writeValueAsString(dest);
            String riskJson = objectMapper.writeValueAsString(risk);
            String confJson = objectMapper.writeValueAsString(conf);

            ContextRecord rec = new ContextRecord(
                    req.request_id(),
                    headersJson,
                    sourceJson,
                    destJson,
                    riskJson,
                    confJson
            );
            recordRepo.save(rec);
        } catch (Exception e) {
            log.warn("Error to persist ContextRecord", e);
        }

        return new EnrichResponse(req.request_id(), source, dest, risk, conf);
    }
}