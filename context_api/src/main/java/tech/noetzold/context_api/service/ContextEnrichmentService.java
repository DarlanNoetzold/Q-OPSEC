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

        try {
            var headersNode = objectMapper.valueToTree(req.headers() != null ? req.headers() : java.util.Map.of());
            var sourceNode = objectMapper.valueToTree(source);
            var destNode   = objectMapper.valueToTree(dest);
            var riskNode   = objectMapper.valueToTree(risk);
            var confNode   = objectMapper.valueToTree(conf);

            ContextRecord rec = new ContextRecord(
                    req.request_id(),
                    headersNode,
                    sourceNode,
                    destNode,
                    riskNode,
                    confNode
            );

            recordRepo.save(rec);
        } catch (Exception e) {
            log.warn("Error to persist ContextRecord", e);
        }

        return new EnrichResponse(req.request_id(), source, dest, risk, conf);
    }
}