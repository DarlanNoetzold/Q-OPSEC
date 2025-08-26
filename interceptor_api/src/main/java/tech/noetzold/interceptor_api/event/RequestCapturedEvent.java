// src/main/java/tech/noetzold/interceptor_api/event/RequestCapturedEvent.java
package tech.noetzold.interceptor_api.event;

import lombok.*;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequestCapturedEvent {
    private String requestId;
    private String method;
    private String uri;
    private String queryString;
    private Map<String, String> headers;
    private String requestBody;
    private long processingTimeMs;
    private long timestamp;
    private String sourceIp;
    private String userAgent;
    private String contentType;
    private int contentLength;
    private String error;
}