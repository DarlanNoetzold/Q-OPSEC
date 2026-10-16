package tech.noetzold.interceptor_api;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import tech.noetzold.interceptor_api.event.RequestCapturedEvent;
import tech.noetzold.interceptor_api.service.AsyncEventService;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class RequestBodyCaptureFilterTest {
    private final AsyncEventService asyncEventService = mock(AsyncEventService.class);

    @Test
    void capturesBodyRedactsSensitiveHeadersAndUsesForwardedClientIp() throws Exception {
        RequestBodyCaptureFilter filter = new RequestBodyCaptureFilter();
        org.springframework.test.util.ReflectionTestUtils.setField(filter, "asyncEventService", asyncEventService);
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/messages");
        request.setContentType("application/json");
        request.setContent("{\"message\":\"hello\"}".getBytes(StandardCharsets.UTF_8));
        request.addHeader("Authorization", "Bearer secret");
        request.addHeader("X-Forwarded-For", "203.0.113.4, 10.0.0.2");
        request.addHeader("User-Agent", "JUnit");
        MockHttpServletResponse response = new MockHttpServletResponse();
        FilterChain chain = (req, res) -> req.getInputStream().readAllBytes();

        filter.doFilter(request, response, chain);

        var captor = org.mockito.ArgumentCaptor.forClass(RequestCapturedEvent.class);
        verify(asyncEventService).publishRequestCaptured(captor.capture());
        RequestCapturedEvent event = captor.getValue();
        assertEquals("POST", event.getMethod());
        assertEquals("/api/messages", event.getUri());
        assertEquals("{\"message\":\"hello\"}", event.getRequestBody());
        assertEquals("[REDACTED]", event.getHeaders().get("Authorization"));
        assertEquals("203.0.113.4", event.getSourceIp());
        assertEquals("JUnit", event.getUserAgent());
        assertEquals(event.getRequestBody().length(), event.getContentLength());
        assertNotNull(event.getRequestId());
        verify(asyncEventService, never()).publishRequestError(any());
    }

    @Test
    void reportsDownstreamFilterFailureAsAnErrorEvent() {
        RequestBodyCaptureFilter filter = new RequestBodyCaptureFilter();
        org.springframework.test.util.ReflectionTestUtils.setField(filter, "asyncEventService", asyncEventService);
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/failure");
        MockHttpServletResponse response = new MockHttpServletResponse();
        FilterChain chain = (req, res) -> { throw new ServletException("downstream failed"); };

        assertDoesNotThrow(() -> filter.doFilter(request, response, chain));

        var captor = org.mockito.ArgumentCaptor.forClass(RequestCapturedEvent.class);
        verify(asyncEventService).publishRequestError(captor.capture());
        assertEquals("downstream failed", captor.getValue().getError());
        verify(asyncEventService, never()).publishRequestCaptured(any());
    }
}
