package tech.noetzold.context_api.controller;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import tech.noetzold.context_api.model.*;
import tech.noetzold.context_api.service.ContextEnrichmentService;
import tech.noetzold.context_api.repository.ContextRecordRepository;

@RestController
@RequestMapping("/context")
public class ContextController {

    private final ContextEnrichmentService service;
    private final ContextRecordRepository recordRepo;

    public ContextController(ContextEnrichmentService service,
                             ContextRecordRepository recordRepo) {
        this.service = service;
        this.recordRepo = recordRepo;
    }

    @PostMapping("/enrich")
    public EnrichResponse enrich(@RequestBody EnrichRequest req, HttpServletRequest httpReq) {
        return service.enrichAll(req, httpReq);
    }

    @PostMapping("/enrich/simple")
    public EnrichResponse enrichSimple(@RequestBody InterceptorPayload payload, HttpServletRequest httpReq) {
        return service.enrichFromInterceptor(payload, httpReq);
    }

    @GetMapping("/record")
    public ResponseEntity<ContextRecord> getRecordByRequestId(@RequestParam("request_id") String requestId) {
        return recordRepo.findByRequestId(requestId)
                .map(ResponseEntity::ok)
                .orElseGet(() -> ResponseEntity.notFound().build());
    }
}