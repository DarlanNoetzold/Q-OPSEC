# Q-OPSEC TECHNICAL DOCUMENTATION

## 1. SYSTEM OVERVIEW
The Q-OPSEC (Quantum-safe Operational Security) project is a framework designed for adaptive cryptography and risk-driven security orchestration. It integrates Reinforcement Learning (RL) agents, Post-Quantum Cryptography (PQC) Key Management, and automated traffic interception to provide a dynamic defense mechanism against evolving threats. The system architecture enforces a pessimistic security posture, prioritizing high-risk signals to determine protective actions.

## 2. INFRASTRUCTURE CONSTRAINTS AND CORE LOGIC
- **Primary Node IP**: 192.168.18.18 (PhD Traceability Network)
- **Security Methodology**: Max-value risk prioritizing (pessimistic logic)
- **Communication Protocol**: RESTful API over HTTP
- **Core Services**: KMS Dispatcher, RL Engine, Classification Agent, Interceptor, Risk Assessor

## 3. MASTER API SPECIFICATION

### MODULE: CRITICAL OPERATIONAL FLOW
#### INTERFACE: 1. Handshake
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:8001/`

#### INTERFACE: 2. KMS Key Gen
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8002/kms/create_key`

#### INTERFACE: 3. Context Enrichment
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8005/context/enrich`

#### INTERFACE: 4. Intercept Analysis
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8080/intercept`

#### INTERFACE: 5. Risk Assessment
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8007/assess`

#### INTERFACE: 6. RL Agent Action
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8089/act`

### MODULE: RECEIVER (INTERNAL INTERCEPTOR)
#### INTERFACE: [POST] /receiver
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8080/receiver`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
  "example": "data"
}
```
- **Technical Implementation Details**: Source: receiver.py

#### INTERFACE: [POST] /decrypt
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:8080/decrypt`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
  "example": "data"
}
```
- **Technical Implementation Details**: Source: receiver.py

#### INTERFACE: [GET] /keys
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:8080/keys`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Technical Implementation Details**: Source: receiver.py

#### INTERFACE: [GET] /logs
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:8080/logs`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Technical Implementation Details**: Source: receiver.py

#### INTERFACE: [GET] /messages
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:8080/messages`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Technical Implementation Details**: Source: receiver.py

#### INTERFACE: [GET] /health
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:8080/health`
- **Required Headers**:
  - `Content-Type`: `application/json`
- **Technical Implementation Details**: Source: receiver.py

### MODULE: CLASSIFICATION_AGENT
#### MODULE: ENDPOINTS
#### INTERFACE: Health check do serviço
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://192.168.18.18:8088/api/v1/health`
- **Required Headers**:
  - `Content-Type`: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: Informações do modelo carregado
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/model`
- **Required Headers**:
  - `Content-Type`: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: /model/reload
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/model/reload`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Request Payload Schema (JSON)**:
```json
{
  "force": 0
}
```
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: Manifest do modelo (schema de entrada/saída)
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/model/manifest`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: /predict
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/predict`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Request Payload Schema (JSON)**:
```json
{
  "data": 0,
  "return_probabilities": 0,
  "send_to_rl": 0,
  "request_id": "string"
}
```
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: Métricas do serviço
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/metrics`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### INTERFACE: Summary de uma sessão específica
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/training/{session_id}/summary`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints.py

#### MODULE: ENDPOINTS_DATASETS
#### INTERFACE: Lista todos os datasets
- **Protocol Method**: `GET`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/datasets`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Technical Implementation Details**: Source: app/api/v1/endpoints_datasets.py

#### INTERFACE: Cria um novo dataset
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:{{port}}/api/v1/datasets`
- **Required Headers**:
  - `Content-Type**: `application/json`
  - `X-API-Key**: `{{api_key}}`
- **Request Payload Schema (JSON)**:
```json
{
  "name": 0,
  "datasets": 0,
  "payload": 0,
  "user": 0,
  "file": 0,
  "override": 0,
  "filename": 0,
  "metadata": 0
}
```
- **Technical Implementation Details**: Source: app/api/v1/endpoints_datasets.py

### MODULE: CONFIABILITY_SERVICE_V2
#### MODULE: TRUST EVALUATION
#### INTERFACE: Evaluate Normal Case
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8083/api/v2/trust/evaluate`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "payload": {
        "claim": "User login attempt",
        "details": {
            "ip": "192.168.18.50",
            "method": "JWT",
            "service": "auth_gate"
        }
    },
    "metadata": {
        "source_id": "internal_gateway",
        "entity_id": "user_admin_01",
        "request_id": "req-norm-001",
        "timestamp": "2026-06-28T10:00:00Z",
        "data_type": "security_event",
        "environment": "production"
    }
}
```

#### INTERFACE: Evaluate Suspicious IP
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8083/api/v2/trust/evaluate`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "payload": {
        "claim": "High volume data transfer",
        "details": {
            "ip": "203.0.113.1",
            "volume_mb": 5000
        }
    },
    "metadata": {
        "source_id": "edge_firewall",
        "entity_id": "unknown_external_host",
        "request_id": "req-susp-002",
        "timestamp": "2026-06-28T10:05:00Z",
        "data_type": "network_traffic",
        "environment": "production"
    }
}
```

### MODULE: CRYPTO_MODULE
#### MODULE: ENCRYPTION
#### INTERFACE: Encrypt Payload
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://{{host}}:{{port}}/encrypt`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "session_id": "sess_abc123xyz",
    "request_id": "req_xyz789abc",
    "plaintext_b64": "SGVsbG8gUUNPUFNFQyE=",
    "algorithm": "AES256_GCM",
    "fetch_from_interceptor": false
}
```
- **Technical Implementation Details**: Encrypts plaintext data using AEAD.

### MODULE: HANDSHAKE_NEGOTIATOR
#### MODULE: NEGOTIATION
#### INTERFACE: Execute Handshake
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8001/handshake`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "proposed": ["KYBER1024", "AES256_GCM"],
    "destination": "http://192.168.18.18:8006/receiver",
    "request_id": "req_{{$guid}}",
    "source": "Postman-Tester",
    "source_id": "origin-alpha"
}
```

### MODULE: INTERCEPTOR_API
#### MODULE: INTERCEPTION
#### INTERFACE: Intercept Payload - Success
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8080/intercept`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "anyRequestId": "QA-INTERCEPT-{{$timestamp}}",
    "sourceIp": "192.168.18.50",
    "destinationIp": "192.168.18.18",
    "payload": "SGVsbG8gUWhEIERhcmxhbiwgUUlTUC1PUFNFQyBJbnRlcmNlcHRpb24gVGVzdA==",
    "protocol": "HTTPS",
    "port": 443
}
```

### MODULE: KMS_SERVICE_V2
#### MODULE: KEY OPERATIONS
#### INTERFACE: Create Key (QKD BB84)
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8002/kms/create_key`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "request_id": "req_{{$guid}}",
    "algorithm": "QKD_BB84",
    "ttl_seconds": 3600
}
```

#### INTERFACE: Create Key (PQC Kyber512)
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8002/kms/create_key`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "request_id": "req_{{$guid}}",
    "algorithm": "Kyber512",
    "ttl_seconds": 3600
}
```

### MODULE: RISK_SERVICE_V2
#### MODULE: PREDICTIONS
#### INTERFACE: Predict Single (Risk Analysis)
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8000/predict/`
- **Request Payload Schema (JSON)**:
```json
{
    "single": {
        "features": {
            "amount": 5000.0,
            "transaction_type": "transfer",
            "source_ip": "192.168.1.50",
            "destination_ip": "10.0.0.1",
            "auth_method": "password",
            "is_new_device": true,
            "classification_agent_label": "top secret"
        }
    },
    "models": ["random_forest", "xgboost"],
    "request_id": "test-123"
}
```

### MODULE: RL_ENGINE
#### MODULE: CORE ACTIONS
#### INTERFACE: Act - High Risk (Should trigger PQC/Hybrid)
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:9009/act`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "request_id": "test-high-{{$guid}}",
    "source": "node-alpha",
    "destination": "node-omega",
    "security_level": "HIGH",
    "risk_score": 0.85,
    "conf_score": 0.90,
    "dst_props": {
        "hardware": ["TPM", "AES-NI"],
        "location_risk": 0.2
    }
}
```

#### INTERFACE: Feedback - Success
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:9009/feedback`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "request_id": "insira-id-aqui",
    "success": true,
    "latency": 120.5,
    "resource_usage": 0.3
}
```

### MODULE: VALIDATION_SEND_API
#### MODULE: VALIDATION & FORWARDING
#### INTERFACE: Send Payload (Normal Flow)
- **Protocol Method**: `POST`
- **Target Endpoint**: `http://192.168.18.18:8005/validation/send`
- **Required Headers**:
  - `Content-Type**: `application/json`
- **Request Payload Schema (JSON)**:
```json
{
    "requestId": "req-{{$guid}}",
    "sessionId": "sess-phd-darlan-v2",
    "selectedAlgorithm": "CRYSTALS-Kyber",
    "cryptoNonceB64": "YWFhYWFhYWFhYWFh",
    "cryptoCiphertextB64": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJi",
    "cryptoAlgorithm": "AES-256-GCM",
    "cryptoExpiresAt": 1720000000,
    "sourceId": "Q-OPSEC-AGENT-01",
    "originUrl": "http://192.168.18.18:3001/webhook-test"
}
```

## 4. OPERATIONAL WORKFLOW
The architectural pipeline follows a discrete six-stage sequential processing model designed for high-assurance environments:
1. **Handshake Protocol**: Initialization of the secure session and identity verification between the client and the orchestrator.
2. **KMS Post-Quantum Key Generation**: Negotiation and generation of NIST-compliant post-quantum cryptographic keys (e.g., Kyber).
3. **Context Enrichment**: Automated gathering of environmental and behavioral metadata (source IP, destination, entity ID) to augment the request context.
4. **Traffic Interception & Analysis**: Deep packet inspection and policy enforcement via the Interceptor API, extracting payloads for secondary analysis.
5. **Pessimistic Risk Assessment**: Calculation of a security score where the highest risk indicator defines the overall security level, rather than an average.
6. **RL-Driven Adaptive Response**: The Reinforcement Learning engine determines and executes the optimal countermeasures (Algorithm selection, Block/Allow, specific QKD/PQC primitives) based on current state.
