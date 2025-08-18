package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.*;

import java.util.List;

@Component
public class ConfidentialityFacade {

    private final ConfidentialityServiceClient confClient;

    public ConfidentialityFacade(ConfidentialityServiceClient confClient) {
        this.confClient = confClient;
    }

    public ContentConfidentiality classify(String reqId,
                                           SourceContext src, DestinationContext dest) {
        return confClient.classify(reqId, src, dest)
                .orElse(new ContentConfidentiality(
                        "internal", 0.4, List.of(), List.of(), List.of(),
                        null, null, "conf-fb-0.0.1"
                ));
    }
}