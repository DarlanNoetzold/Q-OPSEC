package tech.noetzold.context_api.service;

import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class GeoIpService {
    public Optional<String> lookupCountry(String ip) {
        if (ip != null && ip.startsWith("192.168.")) return Optional.of("PrivateNet");
        return Optional.ofNullable(ip).map(x -> "Unknown");
    }
}
