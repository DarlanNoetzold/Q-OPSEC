package tech.noetzold.context_api.service;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Service;
import tech.noetzold.context_api.model.*;

@Service
public class ContextEnrichmentService {

    private final SourceContextBuilder srcBuilder;
    private final DestinationContextBuilder destBuilder;
    private final RiskAssessmentFacade riskFacade;
    private final ConfidentialityFacade confFacade;

    public ContextEnrichmentService(SourceContextBuilder srcBuilder,
                                    DestinationContextBuilder destBuilder,
                                    RiskAssessmentFacade riskFacade,
                                    ConfidentialityFacade confFacade) {
        this.srcBuilder = srcBuilder;
        this.destBuilder = destBuilder;
        this.riskFacade = riskFacade;
        this.confFacade = confFacade;
    }

    public EnrichResponse enrichAll(EnrichRequest req, HttpServletRequest httpReq) {
        SourceContext source = srcBuilder.build(req.source_hint(), req, httpReq);
        DestinationContext dest = destBuilder.build(req.destination_hint(), req);

        RiskContext risk = riskFacade.getRisk(req.request_id(), source, dest);
        ContentConfidentiality conf = confFacade.classify(req.request_id(), source, dest);

        return new EnrichResponse(req.request_id(), source, dest, risk, conf);
    }
}