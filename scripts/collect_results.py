import os
import shutil
from pathlib import Path

BASE_DIR = Path("/home/umbrel/projetos/Q-OPSEC")
OUTPUT_DIR = BASE_DIR / "outputs_unified"

# Map of modules and their metric paths
MODULES = {
    "Risk_Service": BASE_DIR / "risk_service" / "models" / "metrics",
    "Confiability_V1": BASE_DIR / "confiability_service" / "models" / "metrics",
    "Classify_Agent": BASE_DIR / "classify_scheduler" / "models" / "metrics"
}

def collect():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- Q-OPSEC Figures Collector ---")
    
    for mod_name, mod_path in MODULES.items():
        if not mod_path.exists():
            print(f"[!] Path not found for {mod_name}: {mod_path}")
            continue
            
        # Get latest training/retrain folder
        subdirs = [d for d in mod_path.iterdir() if d.is_dir() and (d.name.startswith("training_") or d.name.startswith("retrain_"))]
        if not subdirs:
            print(f"[-] No metric folders found for {mod_name}")
            continue
            
        latest_dir = max(subdirs, key=lambda x: x.name)
        print(f"[*] Found {mod_name} latest: {latest_dir.name}")
        
        target_mod_dir = OUTPUT_DIR / mod_name
        target_mod_dir.mkdir(exist_ok=True)
        
        for img in latest_dir.glob("*.png"):
            shutil.copy2(img, target_mod_dir / img.name)
            print(f"  + Copied {img.name}")

if __name__ == "__main__":
    collect()
