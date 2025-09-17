package tech.noetzold.validation_send_api.controller;


import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;
import tech.noetzold.validation_send_api.model.NegotiationPayload;
import tech.noetzold.validation_send_api.service.ForwardService;

@RestController
@RequestMapping("/validation")
@RequiredArgsConstructor
public class SendController {

    private final ForwardService forwardService;

    @PostMapping("/send")
    public ResponseEntity<?> receiveAndForward(@Valid @RequestBody NegotiationPayload payload) {
        EncryptedDelivery saved = forwardService.receiveAndPersist(payload);
        try {
            forwardService.forwardToOriginOrThrow(payload);
            forwardService.markDelivered(saved);
            return ResponseEntity.ok().body("{\"status\":\"delivered\"}");
        } catch (Exception e) {
            forwardService.markFailed(saved, e.getMessage());
            return ResponseEntity.status(502).body("{\"status\":\"failed\",\"error\":\"" + e.getMessage().replace("\"","'") + "\"}");
        }
    }
}