package tech.noetzold.interceptor_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
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

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

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

    @Value("${classification.api.url:http://127.0.0.1:8088/api/v1/predict}")
    private String classificationApiUrl;

    @Value("${classification.api.key:}")
    private String classificationApiKey;

    @Value("${classification.forward.delay.ms:1000}")
    private long forwardDelayMs;

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

        CompletableFuture.runAsync(() -> {
            try {
                // Delay antes de enviar (não bloqueia a thread do request)
                if (forwardDelayMs > 0) {
                    try {
                        Thread.sleep(forwardDelayMs);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                }

                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(new MediaType("application", "json", StandardCharsets.UTF_8));
                headers.setAccept(MediaType.parseMediaTypes("application/json"));
                headers.setAcceptCharset(java.util.List.of(StandardCharsets.UTF_8));
                headers.set("Connection", "close");
                if (classificationApiKey != null && !classificationApiKey.isEmpty()) {
                    headers.set("X-API-Key", classificationApiKey);
                }
                HttpEntity<Map<String, Object>> httpEntity = new HttpEntity<>(classificationPayload, headers);
                ResponseEntity<String> resp = restTemplate.postForEntity(classificationApiUrl, httpEntity, String.class);
            } catch (HttpStatusCodeException ex) {
                System.err.println("Failed to forward to Classification Agent: " + ex.getRawStatusCode() + " " + ex.getStatusText() + " -> " + ex.getResponseBodyAsString());
            } catch (Exception ex) {
                System.err.println("Failed to forward to Classification Agent: " + ex.getMessage());
            }
        });

        return InterceptResponse.builder()
                .status("OK")
                .requestId(requestId)
                .context(context)
                .build();
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