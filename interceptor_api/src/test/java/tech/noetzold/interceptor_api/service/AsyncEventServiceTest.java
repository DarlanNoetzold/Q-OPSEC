package tech.noetzold.interceptor_api.service;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.event.RequestCapturedEvent;
import tech.noetzold.interceptor_api.model.ContextRequest;

import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class AsyncEventServiceTest {
    private final ContextApiClient contextApiClient = mock(ContextApiClient.class);
    private final AsyncEventService service = new AsyncEventService();

    AsyncEventServiceTest() {
        ReflectionTestUtils.setField(service, "contextApiClient", contextApiClient);
    }

    @Test
    void enrichesNonEmptyCapturedBodyWhenSourceOrDestinationHeaderExists() throws Exception {
        var context = tech.noetzold.interceptor_api.model.ContextResponse.builder().build();
        when(contextApiClient.getContext(any(ContextRequest.class))).thenReturn(context);
        RequestCapturedEvent event = RequestCapturedEvent.builder()
                .requestId("req-1")
                .method("POST")
                .uri("/messages")
                .headers(Map.of("X-Source-Id", "agent-a", "X-Destination-Id", "agent-b"))
                .requestBody("payload")
                .sourceIp("192.0.2.2")
                .userAgent("test-agent")
                .contentType("application/json")
                .processingTimeMs(12)
                .build();

        CompletableFuture<Void> result = service.publishRequestCaptured(event);

        assertNull(result.get());
        var captor = org.mockito.ArgumentCaptor.forClass(ContextRequest.class);
        verify(contextApiClient).getContext(captor.capture());
        assertEquals("agent-a", captor.getValue().getSourceId());
        assertEquals("agent-b", captor.getValue().getDestinationId());
        assertEquals("payload", captor.getValue().getContent());
    }

    @Test
    void skipsContextCallWhenCapturedBodyIsBlank() throws Exception {
        RequestCapturedEvent event = RequestCapturedEvent.builder()
                .requestId("req-2")
                .method("GET")
                .uri("/health")
                .headers(Map.of("X-Source-Id", "agent-a"))
                .requestBody("  ")
                .build();

        service.publishRequestCaptured(event).get();

        verifyNoInteractions(contextApiClient);
    }
}
