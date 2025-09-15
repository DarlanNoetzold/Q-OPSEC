package tech.noetzold.interceptor_api.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import tech.noetzold.interceptor_api.model.*;
import tech.noetzold.interceptor_api.service.InterceptService;

@RestController
@RequestMapping("/intercept")
public class InterceptController {
    @Autowired
    private InterceptService interceptService;

    @PostMapping
    public ResponseEntity<InterceptResponse> intercept(@RequestBody InterceptRequest req) {
        return ResponseEntity.ok(interceptService.intercept(req));
    }

    @GetMapping("/message")
    public ResponseEntity<InterceptedMessageResponse> getMessageByRequestId(@RequestParam("request_id") String requestId) {
        return interceptService.getMessageByRequestId(requestId)
                .map(InterceptedMessageResponse::fromEntity)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }

    // Opcional: compatível com o Crypto Module
    @GetMapping("/payload")
    public ResponseEntity<PayloadResponse> getPayloadByRequestId(@RequestParam("request_id") String requestId) {
        return interceptService.getMessageByRequestId(requestId)
                .map(PayloadResponse::fromEntity)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }
}