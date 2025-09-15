// src/main/java/tech/noetzold/interceptor_api/service/InterceptService.java
package tech.noetzold.interceptor_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.*;
import tech.noetzold.interceptor_api.repository.InterceptedMessageRepository;

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

        Map<String, Object> metadata = new HashMap<>();
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
        return "req_" + System.currentTimeMillis() + "_" +
                UUID.randomUUID().toString().substring(0, 8);
    }
}