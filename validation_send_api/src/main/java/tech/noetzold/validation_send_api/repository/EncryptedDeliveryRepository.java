package tech.noetzold.validation_send_api.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import tech.noetzold.validation_send_api.model.EncryptedDelivery;

import java.util.Optional;

public interface EncryptedDeliveryRepository extends JpaRepository<EncryptedDelivery, Long> {
    Optional<tech.noetzold.validation_send_api.model.EncryptedDelivery> findByRequestId(String requestId);
}