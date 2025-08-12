package tech.noetzold.context_api.repository.impl;

import org.springframework.stereotype.Repository;
import tech.noetzold.context_api.model.DeviceMeta;
import tech.noetzold.context_api.repository.DeviceRepository;

import java.util.Map;
import java.util.Optional;

@Repository
public class InMemoryDeviceRepository implements DeviceRepository {
    private static final Map<String, DeviceMeta> DB = Map.of(
            "dev-123", new DeviceMeta("Android 10", "IoT", "enabled", "compliant")
    );

    @Override
    public Optional<DeviceMeta> findByDeviceId(String deviceId) {
        return Optional.ofNullable(DB.getOrDefault(deviceId, new DeviceMeta("unknown","unknown","unknown","unknown")));
    }
}
