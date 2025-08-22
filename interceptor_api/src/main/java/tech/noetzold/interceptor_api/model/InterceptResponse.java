package tech.noetzold.interceptor_api.model;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InterceptResponse {
    private String status;
    private ContextResponse context;
}
