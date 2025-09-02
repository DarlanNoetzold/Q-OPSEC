package tech.noetzold.interceptor_api.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import tech.noetzold.interceptor_api.model.InterceptRequest;
import tech.noetzold.interceptor_api.model.InterceptResponse;
import tech.noetzold.interceptor_api.service.InterceptService;

@RestController
@RequestMapping("/intercept")
public class InterceptController {
    @Autowired
    private InterceptService interceptService;

    @PostMapping
    public ResponseEntity<InterceptResponse> intercept(@RequestBody InterceptRequest req) {
        return ResponseEntity.ok(interceptService. intercept(req));
    }
}