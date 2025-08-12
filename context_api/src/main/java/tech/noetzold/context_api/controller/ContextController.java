package tech.noetzold.context_api.controller;

import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.web.bind.annotation.*;
import tech.noetzold.context_api.model.EnrichRequest;
import tech.noetzold.context_api.model.EnrichResponse;
import tech.noetzold.context_api.service.ContextEnrichmentService;

@RestController
@RequestMapping("/context")
public class ContextController {

    private final ContextEnrichmentService service;

    public ContextController(ContextEnrichmentService service) {
        this.service = service;
    }

    @PostMapping("/enrich")
    public EnrichResponse enrich(@RequestBody EnrichRequest req, ServerHttpRequest httpReq) {
        return service.enrichAll(req, httpReq);
    }
}