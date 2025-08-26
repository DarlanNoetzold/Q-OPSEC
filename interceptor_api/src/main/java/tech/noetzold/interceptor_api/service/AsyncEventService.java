// src/main/java/tech/noetzold/interceptor_api/service/AsyncEventService.java
package tech.noetzold.interceptor_api.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import tech.noetzold.interceptor_api.event.RequestCapturedEvent;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.ContextRequest;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

@Service
public class AsyncEventService {

    private static final Logger logger = LoggerFactory.getLogger(AsyncEventService.class);

    @Autowired
    private ContextApiClient contextApiClient;

    @Async("taskExecutor")
    public CompletableFuture<Void> publishRequestCaptured(RequestCapturedEvent event) {
        try {
            logger.info("Processing captured request: {} {} from {} - {} bytes",
                    event.getMethod(), event.getUri(), event.getSourceIp(), event.getContentLength());

            analyzeSecurityPatterns(event);

            if (event.getRequestBody() != null && !event.getRequestBody().trim().isEmpty()) {
                enrichWithContextAnalysis(event);
            }

            persistMetrics(event);

        } catch (Exception e) {
            logger.error("Error processing request captured event for {}: {}",
                    event.getRequestId(), e.getMessage(), e);
        }

        return CompletableFuture.completedFuture(null);
    }

    @Async("taskExecutor")
    public CompletableFuture<Void> publishRequestError(RequestCapturedEvent errorEvent) {
        try {
            logger.error("Request processing error: {} {} - {}",
                    errorEvent.getMethod(), errorEvent.getUri(), errorEvent.getError());

            if (isPotentialAttack(errorEvent)) {
                logger.warn("Potential attack detected from {}: {} {}",
                        errorEvent.getSourceIp(), errorEvent.getMethod(), errorEvent.getUri());
            }

        } catch (Exception e) {
            logger.error("Error processing request error event: {}", e.getMessage(), e);
        }

        return CompletableFuture.completedFuture(null);
    }

    private void analyzeSecurityPatterns(RequestCapturedEvent event) {
        String uri = event.getUri().toLowerCase();
        String body = event.getRequestBody();

        if (uri.contains("../") || uri.contains("..\\") ||
                uri.contains("script") || uri.contains("union") ||
                uri.contains("drop") || uri.contains("delete")) {

            logger.warn("Suspicious URI pattern detected: {} from {}",
                    event.getUri(), event.getSourceIp());
        }

        if (body != null) {
            String lowerBody = body.toLowerCase();
            if (lowerBody.contains("<script") || lowerBody.contains("javascript:") ||
                    lowerBody.contains("union select") || lowerBody.contains("drop table")) {

                logger.warn("Suspicious payload detected in request body from {}",
                        event.getSourceIp());
            }
        }

        if (event.getProcessingTimeMs() > 5000) {
            logger.warn("Slow request detected: {}ms for {} from {}",
                    event.getProcessingTimeMs(), event.getUri(), event.getSourceIp());
        }
    }

    private void enrichWithContextAnalysis(RequestCapturedEvent event) {
        try {
            // Extrai informações dos headers para análise de contexto
            String sourceId = event.getHeaders().get("X-Source-Id");
            String destinationId = event.getHeaders().get("X-Destination-Id");

            if (sourceId != null || destinationId != null) {
                Map<String, Object> metadata = new HashMap<>();
                metadata.put("ip", event.getSourceIp());
                metadata.put("userAgent", event.getUserAgent());
                metadata.put("contentType", event.getContentType());
                metadata.put("processingTime", event.getProcessingTimeMs());

                ContextRequest contextReq = ContextRequest.builder()
                        .sourceId(sourceId)
                        .destinationId(destinationId)
                        .content(event.getRequestBody())
                        .metadata(metadata)
                        .build();

                var contextResponse = contextApiClient.getContext(contextReq);

                logger.info("Context analysis completed for request {}: risk={}, confidentiality={}",
                        event.getRequestId(),
                        contextResponse.getRisk(),
                        contextResponse.getConfidentiality());
            }

        } catch (Exception e) {
            logger.error("Error enriching request {} with context: {}",
                    event.getRequestId(), e.getMessage());
        }
    }

    private void persistMetrics(RequestCapturedEvent event) {
        logger.debug("Metrics persisted for request {}", event.getRequestId());
    }

    private boolean isPotentialAttack(RequestCapturedEvent errorEvent) {
        String error = errorEvent.getError();
        if (error == null) return false;

        String lowerError = error.toLowerCase();
        return lowerError.contains("sql") ||
                lowerError.contains("injection") ||
                lowerError.contains("xss") ||
                lowerError.contains("unauthorized") ||
                lowerError.contains("forbidden");
    }
}