package tech.noetzold.interceptor_api;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.*;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Autowired
    private ContextEnrichmentInterceptor contextEnrichmentInterceptor;

    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(contextEnrichmentInterceptor)
                .addPathPatterns("/**")
                .excludePathPatterns(
                        "/health",
                        "/actuator/**"
                );
    }
}