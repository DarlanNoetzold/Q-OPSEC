package tech.noetzold.context_api.repository.impl;

import org.springframework.stereotype.Repository;
import tech.noetzold.context_api.model.DeviceMeta;
import tech.noetzold.context_api.repository.DeviceRepository;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@Repository
public class InMemoryDeviceRepository implements DeviceRepository {
    private static final Map<String, DeviceMeta> DB = new HashMap<>();

    static {


    }

    @Override
    public Optional<DeviceMeta> findByDeviceId(String deviceId) {
        if (deviceId == null || deviceId.isBlank()) {
            return Optional.empty();
        }
        return Optional.ofNullable(DB.get(deviceId));
    }
}