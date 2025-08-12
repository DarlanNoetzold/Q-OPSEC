package tech.noetzold.context_api.service;

import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import tech.noetzold.context_api.model.DestinationContext;
import tech.noetzold.context_api.model.RiskContext;
import tech.noetzold.context_api.model.SourceContext;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Optional;

@Component
public class RiskServiceClient {
    private final WebClient webClient;
    public RiskServiceClient(WebClient riskWebClient) { this.webClient = riskWebClient; }

    public Optional<RiskContext> assess(String requestId, SourceContext src, DestinationContext dst) {
        var payload = Map.of("request_id", requestId, "source", src, "destination", dst);
        try {
            return Optional.ofNullable(
                    webClient.post().uri("/risk/assess")
                            .bodyValue(payload)
                            .retrieve()
                            .bodyToMono(RiskContext.class)
                            .timeout(Duration.ofMillis(200))
                            .onErrorResume(e -> Mono.empty())
                            .block()
            );
        } catch (Exception e) {
            return Optional.of(defaultRisk());
        }
    }

    private RiskContext defaultRisk() {
        return new RiskContext(0.3, "low", 0.1, Map.of("source", "fallback"), 0, java.util.List.of(),
                Instant.now().toString(), "risk-fb-0.0.1");
    }
}