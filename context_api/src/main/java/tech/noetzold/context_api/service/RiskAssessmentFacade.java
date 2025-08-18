package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.*;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Component
public class RiskAssessmentFacade {

    private final RiskServiceClient riskClient;

    public RiskAssessmentFacade(RiskServiceClient riskClient) {
        this.riskClient = riskClient;
    }

    public RiskContext getRisk(String reqId, SourceContext src, DestinationContext dest) {
        return riskClient.assess(reqId, src, dest)
                .orElse(new RiskContext(
                        0.3, "low", 0.1,
                        Map.of("source","fallback"), 0,
                        List.of(), Instant.now().toString(), "risk-fb-0.0.1"
                ));
    }
}