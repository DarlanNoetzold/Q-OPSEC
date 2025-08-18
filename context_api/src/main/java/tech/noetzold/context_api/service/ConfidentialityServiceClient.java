package tech.noetzold.context_api.service;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Component
public class ConfidentialityServiceClient {
    private final WebClient webClient;

    public ConfidentialityServiceClient(@Qualifier("confWebClient") WebClient confWebClient) {
        this.webClient = confWebClient;
    }

    public Optional<ContentConfidentiality> classify(String requestId,
                                                     Map<String, Object> content,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        try {
            Map<String, Object> payload = new HashMap<>();
            if (requestId != null) payload.put("request_id", requestId);
            if (content != null && !content.isEmpty()) payload.put("content", content);
            if (src != null) payload.put("source", src);
            if (dst != null) payload.put("destination", dst);

            ContentConfidentiality resp = webClient.post()
                    .uri("/confidentiality/classify")
                    .bodyValue(payload)
                    .retrieve()
                    .bodyToMono(ContentConfidentiality.class)
                    .timeout(Duration.ofMillis(1000))
                    .onErrorResume(e -> Mono.empty())
                    .block();

            return Optional.ofNullable(resp);
        } catch (Exception e) {
            return Optional.empty();
        }
    }
}