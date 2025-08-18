package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Component
public class ConfidentialityFacade {

    private final ConfidentialityServiceClient confClient;

    public ConfidentialityFacade(ConfidentialityServiceClient confClient) {
        this.confClient = confClient;
    }

    public ContentConfidentiality classify(String reqId,
                                           Map<String, Object> content,
                                           SourceContext src,
                                           DestinationContext dest) {
        Map<String, Object> safeContent = content != null ? content : new HashMap<>();
        return confClient.classify(reqId, safeContent, src, dest)
                .orElse(new ContentConfidentiality(
                        "internal", 0.4, List.of(), List.of(), List.of(),
                        null, null, "conf-fb-0.0.1"
                ));
    }
}