# Q-OPSEC Key Management Service (KMS) v3.0

Serviço de gerenciamento de chaves criptográficas adaptativo para o middleware Q-OPSEC, com suporte a algoritmos clássicos, pós-quânticos (PQC) e Distribuição Quântica de Chaves (QKD).

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI (main.py)                     │
│              Endpoints REST na porta 8002                │
├─────────────────────────────────────────────────────────┤
│                 key_manager.py                           │
│          Orquestração de geração de chaves               │
│          Fallback automático entre camadas               │
├──────────┬──────────────┬───────────────────────────────┤
│ crypto/  │  crypto/     │  crypto/                      │
│classical │  pqc.py      │  quantum.py                   │
│  .py     │              │                               │
│          │ Kyber, NTRU  │  BB84, E91                    │
│ AES, RSA │ SABER, Frodo │  MDI-QKD                      │
│ ECC,     │ McEliece     │                               │
│ ChaCha20 │ HQC, BIKE    │  NetSquid adapter             │
│          │ Dilithium    │  Channel simulator             │
│          │ SPHINCS+     │  Post-processing              │
│          │ Falcon       │                               │
├──────────┴──────────────┴───────────────────────────────┤
│  algorithm_registry.py   │  hardware_profiler.py         │
│  Registro de algoritmos  │  Detecção de hardware         │
│  Cadeia de fallback      │  Benchmarking                 │
│  Classificação           │  Perfis de capacidade         │
├──────────────────────────┴──────────────────────────────┤
│                    storage.py                            │
│              Armazenamento com TTL automático             │
└─────────────────────────────────────────────────────────┘
```

## Algoritmos Suportados

### Clássicos
| Algoritmo | Tipo | Nível de Segurança | Resistente a Quântico |
|-----------|------|--------------------|-----------------------|
| AES256_GCM | Simétrico | Alto | Não |
| AES128_GCM | Simétrico | Médio | Não |
| ChaCha20_Poly1305 | Simétrico | Alto | Não |
| RSA2048 | Assimétrico | Médio | Não |
| RSA4096 | Assimétrico | Alto | Não |
| ECDH_P256 | Assimétrico | Alto | Não |
| ECDH_P384 | Assimétrico | Alto | Não |
| ECDH_P521 | Assimétrico | Alto | Não |

### Pós-Quânticos (PQC) - KEM

| Algoritmo | Família | Nível NIST | Requisitos |
|-----------|---------|------------|------------|
| Kyber512/768/1024 | Lattice | 1/3/5 | Mínimo |
| NTRU-HPS-2048-509/677, NTRU-HPS-4096-821 | Lattice | 1/3/5 | Mínimo |
| NTRU-HRSS-701 | Lattice | 3 | Mínimo |
| LightSaber/Saber/FireSaber | Lattice | 1/3/5 | Mínimo |
| FrodoKEM-640/976/1344-AES | Lattice | 1/3/5 | Médio |
| FrodoKEM-640/976/1344-SHAKE | Lattice | 1/3/5 | Médio |
| Classic-McEliece-348864/460896/6688128/6960119/8192128 | Code-based | 1/3/5 | Alto (RAM) |
| HQC-128/192/256 | Code-based | 1/3/5 | Mínimo |
| BIKE-L1/L3/L5 | Code-based | 1/3/5 | Mínimo |

### Pós-Quânticos (PQC) - Assinaturas

| Algoritmo | Família | Nível NIST |
|-----------|---------|------------|
| Dilithium2/3/5 (ML-DSA) | Lattice | 2/3/5 |
| Falcon-512/1024 | Lattice | 1/5 |
| SPHINCS+-SHA2-128f/192f/256f-simple | Hash-based | 1/3/5 |
| SPHINCS+-SHAKE-128f/256f-simple | Hash-based | 1/5 |

### Quantum Key Distribution (QKD)

| Protocolo | Tipo | Modelo |
|-----------|------|--------|
| QKD_BB84 | Prepare-and-measure | Simulação acadêmica com ruído |
| QKD_E91 | Entanglement-based | Teste de Bell + simulação |
| QKD_MDI | Measurement-device-independent | BSM + canal duplo |
| QKD_CV | Continuous-variable | Mapeado para BB84 |
| QKD_SARG04 | Prepare-and-measure variante | Mapeado para BB84 |
| QKD_DecoyState | Decoy state BB84 | Mapeado para BB84 |
| QKD_DI | Device-independent | Mapeado para E91 |

## Simulação QKD

A simulação QKD implementa modelagem física realista baseada em artigos acadêmicos:

- **Canal quântico**: Atenuação de fibra óptica (0.2 dB/km), eficiência de detector, contagens escuras
- **Modelos de ruído**: Depolarização, erro de fase, desalinhamento óptico, perda de fótons
- **Fonte de fótons**: Distribuição Poissoniana com número médio de fótons configurável
- **Post-processing**: Sifting de bases, correção de erros (Cascade), amplificação de privacidade (SHA3-256)
- **Integração NetSquid**: Adapter para NetSquid quando disponível, com fallback para simulação própria

## Execução

### Local
```bash
pip install -r requirements.txt
python main.py
```

### Docker
```bash
docker build -t qopsec-kms .
docker run -p 8002:8002 -e QKD_AVAILABLE=true qopsec-kms
```

### API
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## Exemplos de Uso

### Criar chave AES-256
```bash
curl -X POST http://localhost:8002/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_001", "algorithm": "AES256_GCM", "ttl_seconds": 3600}'
```

### Criar chave PQC (Kyber)
```bash
curl -X POST http://localhost:8002/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_002", "algorithm": "Kyber768", "ttl_seconds": 7200}'
```

### Criar chave QKD (BB84)
```bash
curl -X POST http://localhost:8002/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_003", "algorithm": "QKD_BB84", "ttl_seconds": 1800}'
```

### Consultar chave
```bash
curl http://localhost:8002/kms/get_key/{session_id}
```

### Listar algoritmos suportados
```bash
curl http://localhost:8002/kms/supported_algorithms
```

### Perfil de hardware
```bash
curl http://localhost:8002/kms/hardware_profile
```

### Health check
```bash
curl http://localhost:8002/health
```

## Requisitos de Hardware por Algoritmo

| Tier | RAM | Cores | Algoritmos Recomendados |
|------|-----|-------|------------------------|
| LOW | < 2 GB | 1-2 | AES, ChaCha20, Kyber512 |
| MEDIUM | 2-8 GB | 2-4 | + Kyber768, NTRU, HQC, BIKE |
| HIGH | 8-64 GB | 4-16 | + FrodoKEM, McEliece, RSA4096, QKD |
| ULTRA | > 64 GB | 16+ | Todos os algoritmos sem restrições |

## Estrutura do Projeto

```
qopsec_kms/
├── main.py                          # FastAPI endpoints
├── models.py                        # Pydantic models
├── key_manager.py                   # Orquestração de geração de chaves
├── storage.py                       # Armazenamento com TTL
├── hardware_profiler.py             # Detecção de hardware e benchmarks
├── algorithm_registry.py            # Registro de algoritmos e fallback
├── crypto/
│   ├── classical.py                 # AES, ChaCha20, RSA, ECC
│   ├── pqc.py                       # Kyber, NTRU, SABER, FrodoKEM, McEliece, HQC, BIKE, Dilithium, SPHINCS+, Falcon
│   └── quantum.py                   # Integração QKD
├── quantum_gateway/
│   ├── netsquid_adapter.py          # Adapter para NetSquid
│   ├── bb84.py                      # Protocolo BB84
│   ├── e91.py                       # Protocolo E91
│   ├── mdi_qkd.py                   # Protocolo MDI-QKD
│   └── channel_simulator.py         # Simulação de canal quântico
├── requirements.txt
├── Dockerfile
└── README.md
```
