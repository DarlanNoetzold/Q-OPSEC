package tech.noetzold.context_api.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import tech.noetzold.context_api.model.ContextRecord;

public interface ContextRecordRepository extends JpaRepository<ContextRecord, Long> {
}