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
                                                     Map<String, Object> contentPointer,
                                                     SourceContext src,
                                                     DestinationContext dst) {
        try {
            Map<String, Object> payload = new HashMap<>();
            if (requestId != null && !requestId.isBlank()) {
                payload.put("request_id", requestId);
            }

            if (contentPointer != null && !contentPointer.isEmpty()) {
                Map<String, Object> cp = new HashMap<>();
                if (contentPointer.get("ref") != null) cp.put("ref", contentPointer.get("ref"));
                if (contentPointer.get("sample_text") != null) cp.put("sample_text", contentPointer.get("sample_text"));
                if (contentPointer.get("metadata") != null) cp.put("metadata", contentPointer.get("metadata"));
                payload.put("content_pointer", cp);
            } else {
                payload.put("content_pointer", new HashMap<>());
            }

            if (src != null && src.ip() != null) {
                Map<String, Object> sourceMin = new HashMap<>();
                sourceMin.put("ip", src.ip());
                payload.put("source", sourceMin);
            }
            if (dst != null && dst.service_id() != null) {
                Map<String, Object> destMin = new HashMap<>();
                destMin.put("service_id", dst.service_id());
                payload.put("destination", destMin);
            }

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