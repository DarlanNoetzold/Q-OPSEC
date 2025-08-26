package tech.noetzold.interceptor_api;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.util.ContentCachingRequestWrapper;
import tech.noetzold.interceptor_api.event.RequestCapturedEvent;
import tech.noetzold.interceptor_api.service.AsyncEventService;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

@Component
@Order(1)
public class RequestBodyCaptureFilter implements Filter {

    private static final Logger logger = LoggerFactory.getLogger(RequestBodyCaptureFilter.class);

    @Autowired
    private AsyncEventService asyncEventService;

    @Override
    public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
            throws IOException, ServletException {

        HttpServletRequest httpRequest = (HttpServletRequest) req;
        ContentCachingRequestWrapper wrapped = new ContentCachingRequestWrapper(httpRequest);

        long startTime = System.currentTimeMillis();
        String requestId = generateRequestId();
        String method = httpRequest.getMethod();
        String uri = httpRequest.getRequestURI();
        String queryString = httpRequest.getQueryString();
        Map<String, String> headers = extractHeaders(httpRequest);

        try {
            chain.doFilter(wrapped, res);

            byte[] contentBytes = wrapped.getContentAsByteArray();
            String requestBody = null;

            if (contentBytes.length > 0) {
                requestBody = new String(contentBytes, StandardCharsets.UTF_8);
                logger.debug("Captured request body for {}: {} bytes", uri, contentBytes.length);
            }

            long processingTime = System.currentTimeMillis() - startTime;

            RequestCapturedEvent event = RequestCapturedEvent.builder()
                    .requestId(requestId)
                    .method(method)
                    .uri(uri)
                    .queryString(queryString)
                    .headers(headers)
                    .requestBody(requestBody)
                    .processingTimeMs(processingTime)
                    .timestamp(System.currentTimeMillis())
                    .sourceIp(getClientIpAddress(httpRequest))
                    .userAgent(httpRequest.getHeader("User-Agent"))
                    .contentType(httpRequest.getContentType())
                    .contentLength(contentBytes.length)
                    .build();

            asyncEventService.publishRequestCaptured(event);

            logger.info("Request captured: {} {} - {} bytes - {}ms",
                    method, uri, contentBytes.length, processingTime);

        } catch (Exception e) {
            logger.error("Error capturing request for {}: {}", uri, e.getMessage(), e);

            RequestCapturedEvent errorEvent = RequestCapturedEvent.builder()
                    .requestId(requestId)
                    .method(method)
                    .uri(uri)
                    .headers(headers)
                    .error(e.getMessage())
                    .timestamp(System.currentTimeMillis())
                    .processingTimeMs(System.currentTimeMillis() - startTime)
                    .build();

            asyncEventService.publishRequestError(errorEvent);
        }
    }

    private String generateRequestId() {
        return "req_" + System.currentTimeMillis() + "_" +
                Integer.toHexString((int)(Math.random() * 0xFFFF));
    }

    private Map<String, String> extractHeaders(HttpServletRequest request) {
        Map<String, String> headers = new HashMap<>();
        Enumeration<String> headerNames = request.getHeaderNames();

        if (headerNames != null) {
            while (headerNames.hasMoreElements()) {
                String headerName = headerNames.nextElement();
                String headerValue = request.getHeader(headerName);

                if (!isSensitiveHeader(headerName)) {
                    headers.put(headerName, headerValue);
                } else {
                    headers.put(headerName, "[REDACTED]");
                }
            }
        }

        return headers;
    }

    private boolean isSensitiveHeader(String headerName) {
        String lowerName = headerName.toLowerCase();
        return lowerName.contains("authorization") ||
                lowerName.contains("cookie") ||
                lowerName.contains("token") ||
                lowerName.contains("password") ||
                lowerName.contains("secret");
    }

    private String getClientIpAddress(HttpServletRequest request) {
        String xForwardedFor = request.getHeader("X-Forwarded-For");
        if (xForwardedFor != null && !xForwardedFor.isEmpty()) {
            return xForwardedFor.split(",")[0].trim();
        }

        String xRealIp = request.getHeader("X-Real-IP");
        if (xRealIp != null && !xRealIp.isEmpty()) {
            return xRealIp;
        }

        return request.getRemoteAddr();
    }
}