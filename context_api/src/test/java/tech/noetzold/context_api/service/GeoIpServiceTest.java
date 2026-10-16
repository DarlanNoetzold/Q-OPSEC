package tech.noetzold.context_api.service;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class GeoIpServiceTest {
    private final GeoIpService service = new GeoIpService();

    @Test
    void identifiesPrivateNetworkAddresses() {
        assertEquals("PrivateNet", service.lookupCountry("192.168.1.2").orElseThrow());
    }

    @Test
    void returnsUnknownForOtherNonNullAddresses() {
        assertEquals("Unknown", service.lookupCountry("203.0.113.2").orElseThrow());
    }

    @Test
    void returnsEmptyForNullAddress() {
        assertTrue(service.lookupCountry(null).isEmpty());
    }
}
