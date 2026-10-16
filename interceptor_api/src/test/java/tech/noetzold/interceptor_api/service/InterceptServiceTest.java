package tech.noetzold.interceptor_api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.client.RestTemplate;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.ContextRequest;
import tech.noetzold.interceptor_api.model.ContextResponse;
import tech.noetzold.interceptor_api.model.InterceptRequest;
import tech.noetzold.interceptor_api.model.InterceptResponse;
import tech.noetzold.interceptor_api.model.InterceptedMessage;
import tech.noetzold.interceptor_api.repository.InterceptedMessageRepository;

import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class InterceptServiceTest {
    @Mock ContextApiClient contextApiClient;
    @Mock InterceptedMessageRepository messageRepo;
    @Mock RestTemplate restTemplate;
    @InjectMocks InterceptService service;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        ReflectionTestUtils.setField(service, "objectMapper", objectMapper);
        ReflectionTestUtils.setField(service, "forwardDelayMs", 0L);
        ReflectionTestUtils.setField(service, "classificationApiUrl", "http://classification.test/predict");
    }

    @Test
    void interceptPersistsMessageEnrichesContextAndReturnsRequestId() {
        java.util.List<InterceptedMessage> savedSnapshots = new java.util.ArrayList<>();
        when(messageRepo.save(any(InterceptedMessage.class))).thenAnswer(invocation -> {
            InterceptedMessage entity = invocation.getArgument(0);
            savedSnapshots.add(InterceptedMessage.builder()
                    .requestId(entity.getRequestId())
                    .message(entity.getMessage())
                    .sourceId(entity.getSourceId())
                    .destinationId(entity.getDestinationId())
                    .metadataJson(entity.getMetadataJson())
                    .status(entity.getStatus())
                    .createdAt(entity.getCreatedAt())
                    .build());
            return entity;
        });
        ContextResponse context = ContextResponse.builder().requestId("context-id").build();
        when(contextApiClient.getContext(any(ContextRequest.class))).thenReturn(context);
        InterceptRequest request = InterceptRequest.builder()
                .message("hello")
                .sourceId("source-1")
                .destinationId("destination-1")
                .metadata(Map.of("channel", "test"))
                .build();

        InterceptResponse response = service.intercept(request);

        assertEquals("OK", response.getStatus());
        assertTrue(response.getRequestId().startsWith("req_"));
        assertSame(context, response.getContext());
        ArgumentCaptor<InterceptedMessage> saved = ArgumentCaptor.forClass(InterceptedMessage.class);
        verify(messageRepo, times(2)).save(saved.capture());
        assertEquals("hello", savedSnapshots.get(0).getMessage());
        assertEquals("intercepted", savedSnapshots.get(0).getStatus());
        assertEquals("processed", savedSnapshots.get(1).getStatus());
        assertEquals(response.getRequestId(), savedSnapshots.get(1).getRequestId());
        ArgumentCaptor<ContextRequest> contextRequest = ArgumentCaptor.forClass(ContextRequest.class);
        verify(contextApiClient).getContext(contextRequest.capture());
        assertEquals("source-1", contextRequest.getValue().getSourceId());
        assertEquals("destination-1", contextRequest.getValue().getDestinationId());
        assertEquals("hello", contextRequest.getValue().getContent());
        assertEquals("test", contextRequest.getValue().getMetadata().get("channel"));
    }

    @Test
    void getMessageByRequestIdDelegatesToRepository() {
        InterceptedMessage message = InterceptedMessage.builder().requestId("req-1").message("hi").build();
        when(messageRepo.findByRequestId("req-1")).thenReturn(Optional.of(message));

        Optional<InterceptedMessage> result = service.getMessageByRequestId("req-1");

        assertEquals(Optional.of(message), result);
        verify(messageRepo).findByRequestId("req-1");
    }
}
