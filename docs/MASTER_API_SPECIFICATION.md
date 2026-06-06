# Q-OPSEC System Architecture & API Specification (Master Reference)

## 1. System Overview
Q-OPSEC is a distributed security cluster designed for **Post-Quantum Cryptography (PQC)** and **AI-Driven Risk Mitigation**. 
The system intercepts standard traffic, negotiates quantum-resistant parameters, manages key lifecycles, and employs Reinforcement Learning (RL) to detect anomalies.

---

## 2. Global Environment
| Variable | Value | Description |
|---|---|---|
| `host` | `192.168.18.18` | Primary Cluster IP |
| `pqc_lib` | `liboqs` | Open Quantum Safe implementation wrapper |

---

## 3. Module Inventory & Endpoint Specification

### 📦 Module: .
#### `POST` /receiver
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `req: Request`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /decrypt
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `req: DecryptRequest`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /keys
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /logs
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `limit: int = 10`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /messages
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `limit: int = 10`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `receiver.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: CLASSIFICATION_AGENT
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /model
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /model/reload
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `padrão`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /model/manifest
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `schema de entrada/saída`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /predict
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `opcional`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /metrics
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/sessions
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `diretórios com training_summary.json`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/latest
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `user: Dict[str, Any] = Depends(get_current_user`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/{session_id}/summary
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `session_id: str,
    user: Dict[str, Any] = Depends(get_current_user`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/images
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `gráficos`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/{session_id}/images/{image_name}
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `gráfico`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /training/latest/images/{image_name}
- **Language**: Python
- **Source File**: `endpoints.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `gráfico`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `soma de todos os arquivos`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /datasets
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `diretório vazio`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `name: str, user=Depends(get_current_user`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `DELETE` /datasets/{name}
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `name: str, user=Depends(require_auth`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /datasets/{name}/files
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``file.filename``
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/files
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/files/{filename}
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `name: str, filename: str, user=Depends(get_current_user`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `DELETE` /datasets/{name}/files/{filename}
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `name: str, filename: str, user=Depends(require_auth`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /datasets/{name}/metadata
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `n/a`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/metadata
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `arquivo metadata.json`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/preview
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `primeiras N linhas`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/schema
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `dtypes`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /datasets/{name}/stats
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `mean, std, min, max, quartis`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /datasets/{name}/validate
- **Language**: Python
- **Source File**: `endpoints_datasets.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `não requeridas`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: CONFIABILITY_SERVICE
#### `GET` /health
- **Language**: Python
- **Source File**: `app.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: CONFIABILITY_SERVICE_V2
#### `GET` /health
- **Language**: Python
- **Source File**: `app.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /
- **Language**: Python
- **Source File**: `app.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: CONTEXT_API
#### `GET` /context/record
- **Language**: Java
- **Source File**: `ContextController.java`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `@RequestParam("requestId"`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /context/enrich
- **Language**: Java
- **Source File**: `ContextController.java`
- **Functional Description**: Context enrichment: appends metadata (Geo, Device, User) to the request.
- **Request Schema/Params**: `@RequestBody EnrichRequest req, HttpServletRequest httpReq`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /context/enrich/simple
- **Language**: Java
- **Source File**: `ContextController.java`
- **Functional Description**: Context enrichment: appends metadata (Geo, Device, User) to the request.
- **Request Schema/Params**: `@RequestBody InterceptorPayload payload, HttpServletRequest httpReq`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: CRYPTO_MODULE
#### `POST` /encrypt
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Data encryption using selected Post-Quantum algorithm (e.g., Kyber).
- **Request Schema/Params**: `n/a`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /encrypt/by-request-id
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Data encryption using selected Post-Quantum algorithm (e.g., Kyber).
- **Request Schema/Params**: `n/a`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /decrypt
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `n/a`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: HANDSHAKE_NEGOTIATOR
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /handshake
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `KMS`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: `n/a`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: INTERCEPTOR_API
#### `GET` /health
- **Language**: Java
- **Source File**: `HealthController.java`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /intercept/message
- **Language**: Java
- **Source File**: `InterceptController.java`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `@RequestParam("request_id"`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /intercept/payload
- **Language**: Java
- **Source File**: `InterceptController.java`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `@RequestParam("request_id"`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: KEY_DESTINATION_ENGINE
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /deliver
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `req: DeliveryRequest`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /delivery/{delivery_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `delivery_id: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /deliveries
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: KMS_SERVICE
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/supported_algorithms
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `ou estrutura`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/algorithm_info/{algorithm}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `ex.: parâmetros, disponibilidade, etc.`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /kms/create_key
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `ou negocia`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/get_key/{session_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `se existir e não estiver expirada`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/get_key
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `FastAPI validation`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/session/{session_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `raw`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `DELETE` /kms/session/{session_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `session_id: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: `liboqs, pqcrypto, etc.`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: KMS_SERVICE_V2
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/supported_algorithms
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/algorithm_info/{algorithm}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `algorithm: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /kms/create_key
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `req: CreateKeyRequest`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/get_key/{key_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `key_id: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/get_key
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `request_id: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `DELETE` /kms/session/{session_id}
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `session_id: str`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /kms/hardware_profile
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /kms/benchmark
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Key Management System operation (PQC Key Generation/Retrieval).
- **Request Schema/Params**: `algorithms: list[str] = None`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: RISK_SERVICE
#### `GET` /health
- **Language**: Python
- **Source File**: `app.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: RISK_SERVICE_V2
#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /summary
- **Language**: Python
- **Source File**: `dataset_info.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `manager: ModelManager = Depends(get_manager`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /versions
- **Language**: Python
- **Source File**: `metrics.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `manager: ModelManager = Depends(get_manager`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /latest
- **Language**: Python
- **Source File**: `metrics.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `manager: ModelManager = Depends(get_manager`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /{version}
- **Language**: Python
- **Source File**: `metrics.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `version: str, manager: ModelManager = Depends(get_manager`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /feature_names
- **Language**: Python
- **Source File**: `models.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `manager: ModelManager = Depends(get_manager`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /feature_names/raw
- **Language**: Python
- **Source File**: `models.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /feature_names/debug_columns
- **Language**: Python
- **Source File**: `models.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /
- **Language**: Python
- **Source File**: `prediction.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `req: PredictRequest, manager: ModelManager = Depends(get_manager`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: RL_ENGINE
#### `POST` /act
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: AI Decision Point: Agent evaluates context and performs security action.
- **Request Schema/Params**: `req: ContextRequest`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /feedback
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `req: FeedbackRequest`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /episode/end
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /metrics
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /training/enable
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /training/disable
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `inference only`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /policy/export
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `path: str = "./exported_policy.json"`
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `POST` /policy/import
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `path: str = "./exported_policy.json"`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /health
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: Health check and service status monitoring.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

#### `GET` /
- **Language**: Python
- **Source File**: `main.py`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: ``
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

### 📦 Module: VALIDATION_SEND_API
#### `POST` /validation/send
- **Language**: Java
- **Source File**: `SendController.java`
- **Functional Description**: General purpose endpoint.
- **Request Schema/Params**: `@Valid @RequestBody NegotiationPayload payload`
- **Typical Payload**: 
```json
{
  "requestId": "string",
  "data": {},
  "metadata": {}
}
```
- **Response**: `200 OK` with JSON payload containing `status`, `timestamp`, and `data` object.

