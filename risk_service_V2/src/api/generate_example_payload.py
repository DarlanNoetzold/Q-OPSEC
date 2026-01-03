import json
from pathlib import Path

FEATURES_PATH_DEFAULT = Path("output/models/v20251231_175538/feature_names.json")

def load_feature_names(path: Path = FEATURES_PATH_DEFAULT):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        if "all_features" in raw and isinstance(raw["all_features"], list):
            return raw["all_features"]
        for key in ("all_features", "numeric_features", "categorical_features"):
            if key in raw and isinstance(raw[key], list):
                return raw[key]
        for v in raw.values():
            if isinstance(v, list):
                return v
        return list(raw.keys())
    if isinstance(raw, list):
        return raw
    return [str(raw)]

def generate_neutral_value(feature_name: str):
    n = feature_name.lower()
    if any(k in n for k in ("amount", "sum", "score", "probability", "mean", "std", "count", "cipher_strength", "security_score", "reputation")):
        return 0.0
    if n.startswith("is_") or n.startswith("has_") or n.endswith("_flag") or n in ("ip_blacklisted", "contains_url", "contains_phone", "is_vpn", "is_proxy", "is_tor", "is_emulator", "is_rooted_or_jailbroken", "is_device_compromised"):
        return False
    if any(k in n for k in ("date", "time", "timezone", "language", "country", "region", "city", "method", "type", "channel", "segment", "vendor", "family", "version")):
        return ""
    if "length" in n or "num_" in n or "devices_last" in n or n.endswith("_count"):
        return 0
    return None

def main(feature_file: str = None, out_file: str = "example_payload.json"):
    path = FEATURES_PATH_DEFAULT if feature_file is None else Path(feature_file)
    features = load_feature_names(path)
    example = {f: generate_neutral_value(f) for f in features}
    payload = {"single": {"features": example}}
    Path(out_file).write_text(json.dumps(payload, indent=2))
    print(f"Example payload written to {out_file}")

if __name__ == "__main__":
    main()