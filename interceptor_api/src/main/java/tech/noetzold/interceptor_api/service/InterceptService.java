package tech.noetzold.interceptor_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.concurrent.ThreadPoolTaskScheduler;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpStatusCodeException;
import org.springframework.web.client.RestTemplate;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.ContextRequest;
import tech.noetzold.interceptor_api.model.ContextResponse;
import tech.noetzold.interceptor_api.model.InterceptRequest;
import tech.noetzold.interceptor_api.model.InterceptResponse;
import tech.noetzold.interceptor_api.model.InterceptedMessage;
import tech.noetzold.interceptor_api.repository.InterceptedMessageRepository;
import tech.noetzold.interceptor_api.resilience.CircuitBreaker;
import tech.noetzold.interceptor_api.resilience.CircuitOpenException;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

@Service
public class InterceptService {

    @Autowired
    private ContextApiClient contextApiClient;

    @Autowired
    private InterceptedMessageRepository messageRepo;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    @Qualifier("forwardScheduler")
    private ThreadPoolTaskScheduler forwardScheduler;

    private final CircuitBreaker classificationBreaker =
            new CircuitBreaker("classification_agent", 5, Duration.ofSeconds(30), 1);

    @Value("${classification.api.url:http://127.0.0.1:8088/api/v1/predict}")
    private String classificationApiUrl;

    @Value("${classification.api.key:}")
    private String classificationApiKey;

    @Value("${classification.forward.delay.ms:1000}")
    private long forwardDelayMs;

    @Value("${classification.forward.max.attempts:3}")
    private int maxForwardAttempts;

    @Value("${classification.forward.retry.base.ms:500}")
    private long forwardRetryBaseMs;

    @Value("${classification.forward.retry.max.ms:5000}")
    private long forwardRetryMaxMs;

    public InterceptResponse intercept(InterceptRequest req) {
        String requestId = generateRequestId();

        InterceptedMessage entity = InterceptedMessage.builder()
                .requestId(requestId)
                .message(req.getMessage())
                .sourceId(req.getSourceId())
                .destinationId(req.getDestinationId())
                .metadataJson(objectMapper.valueToTree(req.getMetadata()))
                .status("intercepted")
                .createdAt(Instant.now())
                .build();

        entity = messageRepo.save(entity);

        ContextRequest ctxReq = new ContextRequest();
        ctxReq.setRequestId(requestId);
        ctxReq.setSourceId(entity.getSourceId());
        ctxReq.setDestinationId(entity.getDestinationId());
        ctxReq.setContent(entity.getMessage());

        Map<String, Object> metadata;
        if (entity.getMetadataJson() != null) {
            try {
                metadata = objectMapper.convertValue(entity.getMetadataJson(), Map.class);
            } catch (Exception e) {
                metadata = req.getMetadata() != null ? req.getMetadata() : new HashMap<>();
            }
        } else {
            metadata = req.getMetadata() != null ? req.getMetadata() : new HashMap<>();
        }
        ctxReq.setMetadata(metadata);

        ContextResponse context = contextApiClient.getContext(ctxReq);

        entity.setStatus("processed");
        messageRepo.save(entity);

        Map<String, Object> classificationPayload = buildPayloadForClassification(entity, context);

        scheduleClassificationForward(classificationPayload, 1);

        return InterceptResponse.builder()
                .status("OK")
                .requestId(requestId)
                .context(context)
                .build();
    }

    private void scheduleClassificationForward(Map<String, Object> payload, int attempt) {
        long delay = attempt <= 1 ? Math.max(0L, forwardDelayMs) : backoffMillis(attempt);
        forwardScheduler.schedule(
                () -> forwardToClassification(payload, attempt),
                Instant.now().plusMillis(delay));
    }

    private void forwardToClassification(Map<String, Object> payload, int attempt) {
        try {
            classificationBreaker.acquire();
        } catch (CircuitOpenException ex) {
            System.err.println("Classification circuit open, dropping forward: " + ex.getMessage());
            return;
        }
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(new MediaType("application", "json", StandardCharsets.UTF_8));
            headers.setAccept(MediaType.parseMediaTypes("application/json"));
            headers.setAcceptCharset(java.util.List.of(StandardCharsets.UTF_8));
            headers.set("Connection", "close");
            if (classificationApiKey != null && !classificationApiKey.isEmpty()) {
                headers.set("X-API-Key", classificationApiKey);
            }
            HttpEntity<Map<String, Object>> httpEntity = new HttpEntity<>(payload, headers);
            ResponseEntity<String> resp = restTemplate.postForEntity(classificationApiUrl, httpEntity, String.class);
            classificationBreaker.recordSuccess();
            System.out.println("Classification forward ok: status=" + resp.getStatusCode().value()
                    + " attempt=" + attempt);
        } catch (HttpStatusCodeException ex) {
            classificationBreaker.recordFailure(ex);
            System.err.println("Failed to forward to Classification Agent: " + ex.getRawStatusCode() + " "
                    + ex.getStatusText() + " -> " + ex.getResponseBodyAsString());
            retryOrGiveUp(payload, attempt, ex);
        } catch (Exception ex) {
            classificationBreaker.recordFailure(ex);
            System.err.println("Failed to forward to Classification Agent: " + ex.getMessage());
            retryOrGiveUp(payload, attempt, ex);
        }
    }

    private void retryOrGiveUp(Map<String, Object> payload, int attempt, Exception error) {
        if (attempt >= maxForwardAttempts) {
            System.err.println("Classification forward gave up after " + attempt + " attempts: "
                    + error.getMessage());
            return;
        }
        scheduleClassificationForward(payload, attempt + 1);
    }

    private long backoffMillis(int attempt) {
        long raw = (long) (forwardRetryBaseMs * Math.pow(2, Math.max(0, attempt - 2)));
        return Math.min(forwardRetryMaxMs, Math.max(1L, raw));
    }

    public Optional<InterceptedMessage> getMessageByRequestId(String requestId) {
        return messageRepo.findByRequestId(requestId);
    }

    private String generateRequestId() {
        return "req_" + System.currentTimeMillis() + "_" + UUID.randomUUID().toString().substring(0, 8);
    }

    private Map<String, Object> buildPayloadForClassification(InterceptedMessage entity, ContextResponse context) {
        Map<String, Object> data = objectMapper.convertValue(context, Map.class);
        if (!data.containsKey("request_id_resolved")) {
            data.put("request_id_resolved", entity.getRequestId());
        }
        if (!data.containsKey("created_at")) {
            data.put("created_at", Instant.now().toString());
        }
        Map<String, Object> body = new HashMap<>();
        body.put("send_to_rl", true);
        body.put("return_probabilities", true);
        body.put("data", data);
        return body;
    }
}