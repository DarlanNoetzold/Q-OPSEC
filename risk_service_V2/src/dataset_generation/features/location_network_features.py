from __future__ import annotations

import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger

logger = get_logger("location_network_features")


def _process_chunk(args):
    """Process a chunk of events in parallel."""
    chunk_indices, ip_ranges = args

    ip_addresses = []
    ip_countries = []
    ip_regions = []
    ip_cities = []
    ip_asns = []
    ip_isps = []
    is_vpn_list = []
    is_datacenter_list = []
    geolocation_lats = []
    geolocation_lons = []

    for idx in chunk_indices:
        ip_range = ip_ranges[idx]

        # Generate synthetic IP
        base_ip = ip_range["range"].split("/")[0]
        octets = base_ip.split(".")
        octets[-1] = str(np.random.randint(1, 254))
        ip = ".".join(octets)

        ip_addresses.append(ip)
        ip_countries.append(ip_range.get("country", "US"))
        ip_regions.append(ip_range.get("region", "Unknown"))
        ip_cities.append(ip_range.get("city", "Unknown"))
        ip_asns.append(ip_range.get("asn", "AS0"))
        ip_isps.append(ip_range.get("isp", "Unknown ISP"))
        is_vpn_list.append(int(ip_range.get("is_vpn", False)))
        is_datacenter_list.append(int(ip_range.get("is_datacenter", False)))
        geolocation_lats.append(ip_range.get("lat", 0.0))
        geolocation_lons.append(ip_range.get("lon", 0.0))

    return {
        "ip_addresses": ip_addresses,
        "ip_countries": ip_countries,
        "ip_regions": ip_regions,
        "ip_cities": ip_cities,
        "ip_asns": ip_asns,
        "ip_isps": ip_isps,
        "is_vpn_list": is_vpn_list,
        "is_datacenter_list": is_datacenter_list,
        "geolocation_lats": geolocation_lats,
        "geolocation_lons": geolocation_lons,
    }


def add_location_network_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.5 Location / Network features (PARALLEL + VECTORIZED)."""

    df = events.copy()
    net_cfg = default_config_loader.load("network_catalog.yaml")

    # Get IP ranges and locations
    ip_ranges = net_cfg.get("ip_ranges", [])
    blacklisted_ips = set(net_cfg.get("blacklisted_ips", []))

    if not ip_ranges:
        logger.warning("No IP ranges found in network_catalog.yaml, using defaults")
        ip_ranges = [
            {
                "range": "192.168.0.0/16",
                "country": "US",
                "region": "California",
                "city": "San Francisco",
                "asn": "AS15169",
                "isp": "Google LLC",
                "lat": 37.7749,
                "lon": -122.4194,
                "is_vpn": False,
                "is_datacenter": False
            }
        ]

    n = len(df)
    num_cores = cpu_count()
    logger.info(f"Assigning IP addresses to {n:,} events using {num_cores} CPU cores...")

    # ✅ VETORIZADO: escolher IP ranges aleatórios para todos os eventos
    chosen_ranges = np.random.choice(len(ip_ranges), size=n)

    # ✅ PARALELISMO: dividir em chunks e processar em paralelo
    chunk_size = max(50_000, n // (num_cores * 4))  # 4 chunks por core
    chunks = []

    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_indices = chosen_ranges[chunk_start:chunk_end]
        chunks.append((chunk_indices, ip_ranges))

    logger.info(f"Processing {len(chunks)} chunks in parallel...")

    # ✅ Processar chunks em paralelo
    with Pool(processes=num_cores) as pool:
        results = pool.map(_process_chunk, chunks)

    # ✅ Combinar resultados
    logger.info("Combining results...")

    ip_addresses = []
    ip_countries = []
    ip_regions = []
    ip_cities = []
    ip_asns = []
    ip_isps = []
    is_vpn_list = []
    is_datacenter_list = []
    geolocation_lats = []
    geolocation_lons = []

    for result in results:
        ip_addresses.extend(result["ip_addresses"])
        ip_countries.extend(result["ip_countries"])
        ip_regions.extend(result["ip_regions"])
        ip_cities.extend(result["ip_cities"])
        ip_asns.extend(result["ip_asns"])
        ip_isps.extend(result["ip_isps"])
        is_vpn_list.extend(result["is_vpn_list"])
        is_datacenter_list.extend(result["is_datacenter_list"])
        geolocation_lats.extend(result["geolocation_lats"])
        geolocation_lons.extend(result["geolocation_lons"])

    # ✅ Atribuir tudo de uma vez
    df["ip_address"] = ip_addresses
    df["ip_country"] = ip_countries
    df["ip_region"] = ip_regions
    df["ip_city"] = ip_cities
    df["ip_asn"] = ip_asns
    df["ip_isp"] = ip_isps
    df["is_vpn"] = is_vpn_list
    df["is_datacenter_ip"] = is_datacenter_list
    df["geolocation_lat"] = geolocation_lats
    df["geolocation_lon"] = geolocation_lons

    # ✅ Blacklist check (vetorizado)
    if blacklisted_ips:
        df["ip_blacklisted"] = df["ip_address"].isin(blacklisted_ips).astype(int)
    else:
        df["ip_blacklisted"] = 0

    # ✅ Proxy/Tor (vetorizado)
    df["is_proxy"] = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    df["is_tor"] = np.random.choice([0, 1], size=n, p=[0.995, 0.005])

    # ✅ Connection type (vetorizado)
    connection_types = ["wifi", "cellular_4g", "cellular_5g", "ethernet", "unknown"]
    connection_weights = [0.5, 0.25, 0.15, 0.08, 0.02]
    df["connection_type"] = np.random.choice(connection_types, size=n, p=connection_weights)

    # ✅ Distance from registered location (vetorizado)
    if "registered_country" in df.columns:
        same_country = df["ip_country"] == df["registered_country"]

        df["is_location_anomaly"] = (~same_country).astype(int)

        df["distance_from_registered_location_km"] = np.where(
            same_country,
            np.random.uniform(0, 500, size=n),
            np.random.uniform(5000, 15000, size=n)
        )
    else:
        df["is_location_anomaly"] = 0
        df["distance_from_registered_location_km"] = 0.0

    logger.info("✅ Location/Network features (1.5) added")
    return df