from __future__ import annotations

import random

import numpy as np
import pandas as pd

from src.common.config_loader import default_config_loader
from src.common.logger import get_logger


logger = get_logger("location_network_features")


def add_location_network_features(events: pd.DataFrame) -> pd.DataFrame:
    """Add 1.5 Location / Network features.

    Fields:
      - ip_address
      - ip_country
      - ip_region
      - ip_city
      - ip_asn
      - ip_isp
      - is_vpn
      - is_proxy
      - is_tor
      - is_datacenter_ip
      - ip_blacklisted
      - geolocation_lat
      - geolocation_lon
      - distance_from_registered_location_km
      - is_location_anomaly
      - connection_type (wifi, cellular, ethernet, etc.)
    """

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

    # Initialize columns
    defaults = {
        "ip_address": None,
        "ip_country": None,
        "ip_region": None,
        "ip_city": None,
        "ip_asn": None,
        "ip_isp": None,
        "is_vpn": 0,
        "is_proxy": 0,
        "is_tor": 0,
        "is_datacenter_ip": 0,
        "ip_blacklisted": 0,
        "geolocation_lat": np.nan,
        "geolocation_lon": np.nan,
        "distance_from_registered_location_km": np.nan,
        "is_location_anomaly": 0,
        "connection_type": "unknown",
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    n = len(df)

    # Assign IP addresses and metadata
    logger.info(f"Assigning IP addresses to {n:,} events...")

    for i in range(n):
        # Pick random IP range
        ip_range = random.choice(ip_ranges)

        # Generate synthetic IP (simplified)
        base_ip = ip_range["range"].split("/")[0]
        octets = base_ip.split(".")
        # Randomize last octet
        octets[-1] = str(random.randint(1, 254))
        ip = ".".join(octets)

        df.at[i, "ip_address"] = ip
        df.at[i, "ip_country"] = ip_range.get("country", "US")
        df.at[i, "ip_region"] = ip_range.get("region", "Unknown")
        df.at[i, "ip_city"] = ip_range.get("city", "Unknown")
        df.at[i, "ip_asn"] = ip_range.get("asn", "AS0")
        df.at[i, "ip_isp"] = ip_range.get("isp", "Unknown ISP")
        df.at[i, "is_vpn"] = int(ip_range.get("is_vpn", False))
        df.at[i, "is_datacenter_ip"] = int(ip_range.get("is_datacenter", False))
        df.at[i, "geolocation_lat"] = ip_range.get("lat", 0.0)
        df.at[i, "geolocation_lon"] = ip_range.get("lon", 0.0)

        # Check blacklist
        if ip in blacklisted_ips:
            df.at[i, "ip_blacklisted"] = 1

    # Proxy/Tor (rare)
    df["is_proxy"] = np.random.choice([0, 1], size=n, p=[0.97, 0.03])
    df["is_tor"] = np.random.choice([0, 1], size=n, p=[0.995, 0.005])

    # Connection type
    connection_types = ["wifi", "cellular_4g", "cellular_5g", "ethernet", "unknown"]
    connection_weights = [0.5, 0.25, 0.15, 0.08, 0.02]
    df["connection_type"] = random.choices(connection_types, weights=connection_weights, k=n)

    # Distance from registered location (if registered_country exists)
    if "registered_country" in df.columns:
        # Simplified: if IP country != registered country, mark as anomaly
        df["is_location_anomaly"] = (
            df["ip_country"] != df["registered_country"]
        ).astype(int)

        # Synthetic distance (0-5000 km for same country, 5000-15000 for different)
        same_country = df["ip_country"] == df["registered_country"]
        df.loc[same_country, "distance_from_registered_location_km"] = np.random.uniform(0, 500, size=same_country.sum())
        df.loc[~same_country, "distance_from_registered_location_km"] = np.random.uniform(5000, 15000, size=(~same_country).sum())

    logger.info("Location/Network features (1.5) added")
    return df