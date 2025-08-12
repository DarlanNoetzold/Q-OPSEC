package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.context_api.model.ContentConfidentiality;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.SourceContext;

import java.time.Duration;
import java.util.Map;
import java.util.Optional;

@Component
public class ConfidentialityServiceClient {
    private final WebClient webClient;
    public ConfidentialityServiceClient(WebClient confWebClient) { this.webClient = confWebClient; }

    public Optional<ContentConfidentiality> classify(String requestId, Map<String,Object> contentPointer,
                                                     SourceContext src, DestinationContext dst) {
        var payload = Map.of(
                "request_id", requestId,
                "content_pointer", contentPointer,
                "source", src,
                "destination", dst
        );
        try {
            return Optional.ofNullable(
                    webClient.post().uri("/confidentiality/classify")
                            .bodyValue(payload)
                            .retrieve()
                            .bodyToMono(ContentConfidentiality.class)
                            .timeout(Duration.ofMillis(250))
                            .onErrorResume(e -> Mono.empty())
                            .block()
            );
        } catch (Exception e) {
            return Optional.empty();
        }
    }
}
