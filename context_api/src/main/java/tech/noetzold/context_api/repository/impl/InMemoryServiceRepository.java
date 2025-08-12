package tech.noetzold.context_api.repository.impl;

import org.springframework.stereotype.Repository;
import tech.noetzold.context_api.model.ServiceMeta;
import tech.noetzold.context_api.repository.ServiceRepository;

import java.util.List;
import java.util.Optional;

@Repository
public class InMemoryServiceRepository implements ServiceRepository {

    @Override
    public Optional<ServiceMeta> findByServiceIdAndHost(String serviceId, String host, String path) {
        // Stubbed example
        return Optional.of(new ServiceMeta(
                "10.0.0.5",
                "database",
                "high",
                "monitored",
                "Linux 5.10",
                List.of("TLS1.3")
        ));
    }
}