package tech.noetzold.interceptor_api;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.*;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.util.List;

@Configuration
public class RestConfig {

    @Bean
    public RestTemplate restTemplate() {
        var factory = new HttpComponentsClientHttpRequestFactory();
        factory.setConnectTimeout((int) Duration.ofSeconds(3).toMillis());
        factory.setReadTimeout((int) Duration.ofSeconds(5).toMillis());

        RestTemplate rt = new RestTemplate(factory);
        rt.setInterceptors(List.of((request, body, execution) -> {
            request.getHeaders().add("X-Interceptor", "interceptor_api");
            return execution.execute(request, body);
        }));
        return rt;
    }
}