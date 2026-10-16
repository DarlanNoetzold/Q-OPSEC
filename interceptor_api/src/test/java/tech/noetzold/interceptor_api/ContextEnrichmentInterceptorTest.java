package tech.noetzold.interceptor_api;

import jakarta.servlet.http.HttpServletResponse;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.ContextRequest;
import tech.noetzold.interceptor_api.model.ContextResponse;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class ContextEnrichmentInterceptorTest {
    private final ContextApiClient client = mock(ContextApiClient.class);
    private final ContextEnrichmentInterceptor interceptor = new ContextEnrichmentInterceptor(client);

    @Test
    void addsEnrichedContextFromRequestHeaders() throws Exception {
        ContextResponse context = ContextResponse.builder().risk(Map.of("level", "low")).build();
        when(client.getContext(any(ContextRequest.class))).thenReturn(context);
        MockHttpServletRequest request = new MockHttpServletRequest();
        request.addHeader("X-Source-Id", "source-a");
        request.addHeader("X-Destination-Id", "destination-b");

        boolean continueChain = interceptor.preHandle(request, new MockHttpServletResponse(), new Object());

        assertTrue(continueChain);
        assertSame(context, request.getAttribute("context.enriched"));
        assertNull(request.getAttribute("context.error"));
        var captor = org.mockito.ArgumentCaptor.forClass(ContextRequest.class);
        verify(client).getContext(captor.capture());
        assertEquals("source-a", captor.getValue().getSourceId());
        assertEquals("destination-b", captor.getValue().getDestinationId());
        assertNull(captor.getValue().getContent());
        assertTrue(captor.getValue().getMetadata().isEmpty());
    }

    @Test
    void recordsContextFailureAndStillAllowsRequestToProceed() throws Exception {
        when(client.getContext(any(ContextRequest.class))).thenThrow(new IllegalStateException("context unavailable"));
        MockHttpServletRequest request = new MockHttpServletRequest();

        boolean continueChain = interceptor.preHandle(request, mock(HttpServletResponse.class), new Object());

        assertTrue(continueChain);
        assertEquals("context unavailable", request.getAttribute("context.error"));
        assertNull(request.getAttribute("context.enriched"));
    }
}
