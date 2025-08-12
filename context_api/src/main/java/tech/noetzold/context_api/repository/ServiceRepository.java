package tech.noetzold.context_api.repository;

import tech.noetzold.context_api.model.ServiceMeta;

import java.util.Optional;

public interface ServiceRepository {
    Optional<ServiceMeta> findByServiceIdAndHost(String serviceId, String host, String path);
}
