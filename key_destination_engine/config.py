import os

# Server configuration
HOST = os.getenv("KDE_HOST", "0.0.0.0")
PORT = int(os.getenv("KDE_PORT", "8003"))

# Delivery timeouts
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MQTT_TIMEOUT = int(os.getenv("MQTT_TIMEOUT", "10"))
HSM_TIMEOUT = int(os.getenv("HSM_TIMEOUT", "60"))

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# File delivery paths
FILE_DELIVERY_BASE_PATH = os.getenv("FILE_DELIVERY_PATH", "./delivered_keys")

# HSM Configuration
HSM_LIBRARY_PATH = os.getenv("HSM_LIBRARY_PATH", "/usr/lib/libpkcs11.so")
HSM_SLOT_ID = int(os.getenv("HSM_SLOT_ID", "0"))

# Supported methods (canonical)
SUPPORTED_METHODS = ["API", "MQTT", "HSM", "FILE"]