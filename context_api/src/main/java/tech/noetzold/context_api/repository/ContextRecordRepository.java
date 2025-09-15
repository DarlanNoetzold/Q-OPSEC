package tech.noetzold.context_api.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import tech.noetzold.context_api.model.ContextRecord;

import java.util.Optional;

public interface ContextRecordRepository extends JpaRepository<ContextRecord, Long> {
    Optional<ContextRecord> findByRequestId(String requestId);
}