package tech.noetzold.interceptor_api;

import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import tech.noetzold.interceptor_api.client.ContextApiClient;
import tech.noetzold.interceptor_api.model.ContextRequest;
import tech.noetzold.interceptor_api.model.ContextResponse;

import java.util.Collections;

@Component
public class ContextEnrichmentInterceptor implements HandlerInterceptor {

    private final ContextApiClient contextApiClient;

    public ContextEnrichmentInterceptor(ContextApiClient contextApiClient) {
        this.contextApiClient = contextApiClient;
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        try {
            String sourceId = request.getHeader("X-Source-Id");
            String destinationId = request.getHeader("X-Destination-Id");

            ContextRequest ctxReq = ContextRequest.builder()
                    .sourceId(sourceId)
                    .destinationId(destinationId)
                    .content(null)
                    .metadata(Collections.emptyMap())
                    .build();

            ContextResponse ctx = contextApiClient.getContext(ctxReq);
            request.setAttribute("context.enriched", ctx);
        } catch (Exception e) {
            request.setAttribute("context.error", e.getMessage());
        }
        return true;
    }
}