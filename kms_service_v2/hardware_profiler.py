import os
import time
import platform
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

import psutil

logger = logging.getLogger("qopsec.hardware")


class CapabilityTier(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    ULTRA = "ULTRA"


class AlgorithmAvailability(Enum):
    AVAILABLE = "AVAILABLE"
    SLOW = "SLOW"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class CPUProfile:
    architecture: str = ""
    model_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    base_frequency_mhz: float = 0.0
    max_frequency_mhz: float = 0.0
    cache_l1_bytes: int = 0
    cache_l2_bytes: int = 0
    cache_l3_bytes: int = 0
    has_aes_ni: bool = False
    has_avx: bool = False
    has_avx2: bool = False
    has_avx512: bool = False
    has_sha_extensions: bool = False
    has_rdrand: bool = False
    has_sse42: bool = False


@dataclass
class MemoryProfile:
    total_ram_bytes: int = 0
    available_ram_bytes: int = 0
    swap_total_bytes: int = 0
    swap_available_bytes: int = 0

    @property
    def total_ram_gb(self) -> float:
        return self.total_ram_bytes / (1024 ** 3)

    @property
    def available_ram_gb(self) -> float:
        return self.available_ram_bytes / (1024 ** 3)


@dataclass
class GPUProfile:
    detected: bool = False
    model: str = ""
    vram_bytes: int = 0
    cuda_available: bool = False
    opencl_available: bool = False

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)


@dataclass
class BenchmarkResult:
    algorithm: str = ""
    keygen_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    memory_usage_bytes: int = 0
    availability: AlgorithmAvailability = AlgorithmAvailability.AVAILABLE


@dataclass
class HardwareProfile:
    cpu: CPUProfile = field(default_factory=CPUProfile)
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    gpu: GPUProfile = field(default_factory=GPUProfile)
    capability_tier: CapabilityTier = CapabilityTier.MEDIUM
    benchmarks: Dict[str, BenchmarkResult] = field(default_factory=dict)
    qkd_hardware_detected: bool = False
    profiled_at: float = 0.0


def detect_cpu() -> CPUProfile:
    profile = CPUProfile()
    profile.architecture = platform.machine()
    profile.physical_cores = psutil.cpu_count(logical=False) or 1
    profile.logical_cores = psutil.cpu_count(logical=True) or 1

    try:
        freq = psutil.cpu_freq()
        if freq:
            profile.base_frequency_mhz = freq.current or 0.0
            profile.max_frequency_mhz = freq.max or freq.current or 0.0
    except Exception:
        pass

    profile.model_name = _read_cpu_model_name()
    _detect_cpu_caches(profile)
    _detect_cpu_features(profile)

    return profile


def _read_cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    return platform.processor() or "unknown"


def _detect_cpu_caches(profile: CPUProfile):
    try:
        cache_base = "/sys/devices/system/cpu/cpu0/cache"
        if not os.path.exists(cache_base):
            return

        for index_dir in sorted(os.listdir(cache_base)):
            index_path = os.path.join(cache_base, index_dir)
            level_file = os.path.join(index_path, "level")
            size_file = os.path.join(index_path, "size")

            if not os.path.exists(level_file) or not os.path.exists(size_file):
                continue

            with open(level_file) as f:
                level = int(f.read().strip())
            with open(size_file) as f:
                size_str = f.read().strip()

            size_bytes = _parse_cache_size(size_str)

            if level == 1:
                profile.cache_l1_bytes = max(profile.cache_l1_bytes, size_bytes)
            elif level == 2:
                profile.cache_l2_bytes = max(profile.cache_l2_bytes, size_bytes)
            elif level == 3:
                profile.cache_l3_bytes = max(profile.cache_l3_bytes, size_bytes)
    except Exception:
        pass


def _parse_cache_size(size_str: str) -> int:
    size_str = size_str.upper().strip()
    if size_str.endswith("K"):
        return int(size_str[:-1]) * 1024
    if size_str.endswith("M"):
        return int(size_str[:-1]) * 1024 * 1024
    if size_str.endswith("G"):
        return int(size_str[:-1]) * 1024 * 1024 * 1024
    return int(size_str)


def _detect_cpu_features(profile: CPUProfile):
    flags = _read_cpu_flags()

    profile.has_aes_ni = "aes" in flags
    profile.has_avx = "avx" in flags
    profile.has_avx2 = "avx2" in flags
    profile.has_avx512 = any(f.startswith("avx512") for f in flags)
    profile.has_sha_extensions = "sha_ni" in flags or "sha" in flags
    profile.has_rdrand = "rdrand" in flags
    profile.has_sse42 = "sse4_2" in flags


def _read_cpu_flags() -> set:
    flags = set()
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    flag_str = line.split(":", 1)[1].strip()
                    flags = set(flag_str.split())
                    break
    except (FileNotFoundError, PermissionError):
        pass
    return flags


def detect_memory() -> MemoryProfile:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemoryProfile(
        total_ram_bytes=vm.total,
        available_ram_bytes=vm.available,
        swap_total_bytes=swap.total,
        swap_available_bytes=swap.free,
    )


def detect_gpu() -> GPUProfile:
    profile = GPUProfile()

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            profile.detected = True
            profile.model = parts[0].strip()
            if len(parts) > 1:
                profile.vram_bytes = int(float(parts[1].strip()) * 1024 * 1024)
            profile.cuda_available = True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    if not profile.detected:
        try:
            import subprocess
            result = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                profile.opencl_available = True
        except (FileNotFoundError, Exception):
            pass

    return profile


def determine_capability_tier(cpu: CPUProfile, memory: MemoryProfile,
                               gpu: GPUProfile) -> CapabilityTier:
    score = 0

    score += min(cpu.physical_cores, 16) * 2
    score += min(cpu.logical_cores, 32)

    if cpu.max_frequency_mhz > 3500:
        score += 10
    elif cpu.max_frequency_mhz > 2500:
        score += 5

    if cpu.has_aes_ni:
        score += 5
    if cpu.has_avx2:
        score += 5
    if cpu.has_avx512:
        score += 10

    ram_gb = memory.total_ram_bytes / (1024 ** 3)
    if ram_gb >= 64:
        score += 20
    elif ram_gb >= 16:
        score += 10
    elif ram_gb >= 8:
        score += 5

    if gpu.detected and gpu.cuda_available:
        score += 15

    if score >= 60:
        return CapabilityTier.ULTRA
    if score >= 35:
        return CapabilityTier.HIGH
    if score >= 15:
        return CapabilityTier.MEDIUM
    return CapabilityTier.LOW


def run_algorithm_benchmarks(algorithms: Optional[List[str]] = None) -> Dict[str, BenchmarkResult]:
    from crypto.classical import is_classical_algorithm, generate_classical_key
    from crypto.pqc import is_pqc_algorithm, generate_pqc_key

    if algorithms is None:
        algorithms = [
            "AES256_GCM", "AES128_GCM", "ChaCha20_Poly1305",
            "RSA2048", "RSA4096", "ECDH_P256",
            "Kyber512", "Kyber768", "Kyber1024",
            "Dilithium2", "Dilithium3",
            "SPHINCS+-SHA2-128f-simple",
            "FrodoKEM-640-AES", "Classic-McEliece-348864",
            "HQC-128", "BIKE-L1",
        ]

    results = {}
    iterations = 5

    for algo in algorithms:
        benchmark = BenchmarkResult(algorithm=algo)
        timings = []

        for _ in range(iterations):
            try:
                start = time.perf_counter()
                if is_classical_algorithm(algo):
                    generate_classical_key(algo)
                elif is_pqc_algorithm(algo):
                    generate_pqc_key(algo)
                elapsed = time.perf_counter() - start
                timings.append(elapsed)
            except Exception:
                benchmark.availability = AlgorithmAvailability.UNAVAILABLE
                break

        if timings:
            avg_time = sum(timings) / len(timings)
            benchmark.keygen_time_ms = avg_time * 1000
            benchmark.throughput_ops_per_sec = 1.0 / avg_time if avg_time > 0 else 0

            if avg_time > 5.0:
                benchmark.availability = AlgorithmAvailability.SLOW
            elif avg_time > 30.0:
                benchmark.availability = AlgorithmAvailability.UNAVAILABLE

        results[algo] = benchmark

    return results


def get_algorithm_availability(algorithm: str, profile: HardwareProfile) -> AlgorithmAvailability:
    if algorithm in profile.benchmarks:
        return profile.benchmarks[algorithm].availability

    mceliece_names = ("Classic-McEliece", "McEliece")
    if any(algorithm.startswith(prefix) for prefix in mceliece_names):
        if profile.memory.available_ram_gb < 2.0:
            return AlgorithmAvailability.UNAVAILABLE
        if profile.capability_tier == CapabilityTier.LOW:
            return AlgorithmAvailability.SLOW

    frodo_names = ("FrodoKEM",)
    if any(algorithm.startswith(prefix) for prefix in frodo_names):
        if profile.capability_tier == CapabilityTier.LOW:
            return AlgorithmAvailability.SLOW

    return AlgorithmAvailability.AVAILABLE


def suggest_alternative(algorithm: str, profile: HardwareProfile) -> Optional[str]:
    alternatives = {
        "RSA4096": "ECDH_P256",
        "Classic-McEliece-8192128": "Kyber1024",
        "Classic-McEliece-6960119": "Kyber1024",
        "Classic-McEliece-6688128": "Kyber768",
        "Classic-McEliece-460896": "Kyber768",
        "Classic-McEliece-348864": "Kyber512",
        "FrodoKEM-1344-AES": "Kyber1024",
        "FrodoKEM-976-AES": "Kyber768",
        "FrodoKEM-640-AES": "Kyber512",
        "SPHINCS+-SHA2-256f-simple": "Dilithium5",
        "SPHINCS+-SHA2-192f-simple": "Dilithium3",
        "SPHINCS+-SHA2-128f-simple": "Dilithium2",
    }
    return alternatives.get(algorithm)


_cached_profile: Optional[HardwareProfile] = None
_cache_ttl_seconds = 300


def get_hardware_profile(force_refresh: bool = False) -> HardwareProfile:
    global _cached_profile

    if not force_refresh and _cached_profile:
        if time.time() - _cached_profile.profiled_at < _cache_ttl_seconds:
            return _cached_profile

    cpu = detect_cpu()
    memory = detect_memory()
    gpu = detect_gpu()
    tier = determine_capability_tier(cpu, memory, gpu)

    profile = HardwareProfile(
        cpu=cpu,
        memory=memory,
        gpu=gpu,
        capability_tier=tier,
        qkd_hardware_detected=os.getenv("QKD_AVAILABLE", "true").lower() == "true",
        profiled_at=time.time(),
    )

    _cached_profile = profile
    return profile


def get_hardware_summary() -> Dict[str, Any]:
    profile = get_hardware_profile()
    return {
        "capability_tier": profile.capability_tier.value,
        "cpu": {
            "architecture": profile.cpu.architecture,
            "model": profile.cpu.model_name,
            "physical_cores": profile.cpu.physical_cores,
            "logical_cores": profile.cpu.logical_cores,
            "max_frequency_mhz": profile.cpu.max_frequency_mhz,
            "cache_l3_bytes": profile.cpu.cache_l3_bytes,
            "aes_ni": profile.cpu.has_aes_ni,
            "avx2": profile.cpu.has_avx2,
            "avx512": profile.cpu.has_avx512,
            "sha_extensions": profile.cpu.has_sha_extensions,
            "rdrand": profile.cpu.has_rdrand,
        },
        "memory": {
            "total_ram_gb": round(profile.memory.total_ram_gb, 2),
            "available_ram_gb": round(profile.memory.available_ram_gb, 2),
        },
        "gpu": {
            "detected": profile.gpu.detected,
            "model": profile.gpu.model,
            "cuda_available": profile.gpu.cuda_available,
        },
        "qkd_hardware": profile.qkd_hardware_detected,
    }
