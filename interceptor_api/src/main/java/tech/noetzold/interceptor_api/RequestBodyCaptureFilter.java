package tech.noetzold.interceptor_api;

import org.springframework.stereotype.Component;
import org.springframework.web.util.ContentCachingRequestWrapper;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Component
public class RequestBodyCaptureFilter implements Filter {

    @Override
    public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
            throws IOException, ServletException {

        ContentCachingRequestWrapper wrapped = new ContentCachingRequestWrapper((HttpServletRequest) req);
        chain.doFilter(wrapped, res);

        byte[] buf = wrapped.getContentAsByteArray();
        if (buf.length > 0) {
            String body = new String(buf, StandardCharsets.UTF_8);
            req.setAttribute("rawBody", body);
        }
    }
}
