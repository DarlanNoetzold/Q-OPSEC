package tech.noetzold.interceptor_api.client;

import tech.noetzold.interceptor_api.model.ContextRequest;
import tech.noetzold.interceptor_api.model.ContextResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class ContextApiClient {

    private final RestTemplate restTemplate;

    public ContextApiClient(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${context.api.url:http://localhost:8081/context/enrich/simple}")
    private String contextApiUrl;

    public ContextResponse getContext(ContextRequest req) {
        return restTemplate.postForObject(contextApiUrl, req, ContextResponse.class);
    }
}