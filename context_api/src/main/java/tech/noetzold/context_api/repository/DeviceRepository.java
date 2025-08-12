package tech.noetzold.context_api.repository;

import java.util.Optional;

public interface DeviceRepository {
    Optional<tech.noetzold.context_api.model.DeviceMeta> findByDeviceId(String deviceId);
}
