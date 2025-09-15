package tech.noetzold.interceptor_api.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import tech.noetzold.interceptor_api.model.InterceptedMessage;

import java.util.Optional;

public interface InterceptedMessageRepository extends JpaRepository<InterceptedMessage, Long> {
    Optional<InterceptedMessage> findByRequestId(String requestId);
}