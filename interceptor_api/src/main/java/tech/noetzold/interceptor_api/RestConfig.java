package tech.noetzold.interceptor_api;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

@Configuration
public class RestConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofSeconds(3))
                .setReadTimeout(Duration.ofSeconds(5))
                .additionalInterceptors((request, body, execution) -> {
                    request.getHeaders().add("X-Interceptor", "interceptor_api");
                    return execution.execute(request, body);
                })
                .build();
    }
}