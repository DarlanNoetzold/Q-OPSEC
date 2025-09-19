### O que já foi feito/O que falta:

### **interceptor_api**
- **Intercept** → receive messages via REST endpoint
- **Extract** → gather all metadata
- **Forward** → send enriched data to Context_API
- **Persist** → save intercepted message in database
- **Retrieve** → provide endpoint to fetch message by request ID

---
### **context_api**
- **Build Source** → create source context
- **Build Destination** → create destination context
- **Risk** → call Risk service to compute risk context
- **Reliability** → call confiability service for conf context
- **Assemble** → provide endpoint to build full context payload
- **Query** → provide endpoint to retrieve stored context payload

---
### **risk_api**
- **Train** → train risk assessment models with incoming payload
- **Assess** → evaluate risk for given context payload (returns 503 if model not ready)
- **Cleanup All** → remove all trained risk models, reset registry and service state
- **Cleanup Selective** → clean up old/unused/low-performing risk models (keep N best, max age, accuracy threshold)
- **Recommendations** → provide cleanup recommendations (based on size, age, accuracy, f1 score)
- **Scheduled Maintenance** → supports periodic retraining and model cleanup strategy

---
### **confidentiality_api**
- **Train** → train confidentiality models from synthetic or provided data
- **Classify** → classify text into confidentiality levels (public, internal, confidential, restricted)
- **Health** → health-check endpoint for service status
- **Cleanup All** → delete all models, reset registry and service state
- **Cleanup Selective** → remove old/low-accuracy models based on rules (keep N best, age, accuracy)
- **Recommendations** → suggest cleanup strategy (keep_n, max_age, accuracy threshold)
- **Scheduled Retrain & Cleanup** → background retraining and automatic model maintenance

---
### **load_intercept** (Load Testing Tool)
- **Generate** → create diverse synthetic payloads (public, internal, confidential, restricted)
- **Load Test** → send concurrent HTTP requests to `/intercept` endpoint with controlled RPS
- **Simulate** → mimic real traffic with varied metadata (doc_types, apps, severities, regions)
- **Control** → configurable concurrency, total requests, timeout, and rate limiting
- **Monitor** → track progress, success/error rates, and performance metrics
- **Randomize** → use seeded randomization for reproducible test scenarios

---
### **dataset_procedure** (dataset_extractor - V1)
TO-DO - Get data from context_db, process it into a time series and create a dataset

---
### **classify_scheduler**
TO-DO - Train models to classify security level

---
### **classify_agent**
TO-DO - Uses the trained models, takes the new information and classifies the security level

---
### **rl_engine**
TO-DO - Models an RL so that with the contexts and security level it is possible to define combinations of own algorithms, with or without the possibility of quantum algorithms, or post-quantum or hybrid models

---
### **kms_service**
- **Key Lifecycle** → create, rotate, expire and revoke cryptographic keys
- **Key Registry** → track and manage key metadata (owner, validity, policies)
- **Cache & Storage** → maintain in‑memory cache and persistent storage for keys
- **Access Control** → expose controlled key retrieval/rotation APIs (no data encryption itself)
- **Integration** → communicates only via Quantivco API Gateway for secure access

---
### **quantum_gateway**
- **QKD Integration** → interfaces with **Quantum Key Distribution (QKD) hardware** via `qkd_interface.py`
- **Hardware Detection** → automatically detects available quantum hardware/providers (`hardware_detector.py`)
- **Compatibility Layer** → provides a unified API regardless of vendor‑specific protocols (`compatibility_layer.py`)
- **Fallback Management** → switches to classical key exchange (e.g., PKI/KMS) if QKD is unavailable (`fallback_manager.py`)
- **Key Delivery** → securely delivers quantum‑safe keys to the **KMS Service** through controlled channels
- **Resilience** → maintains transparent failover between quantum hardware and software fallback
- **Integration** → acts as the **bridge between external quantum infra** and internal services (via Quantivco API Gateway)

---
### **handshake_negociator**
- **Negotiation Engine** → implements secure key exchange and handshake logic (`negotiator.py`)
- **Policy Enforcement** → applies cryptographic/security policies from `policies.yaml`
- **Modeling** → request/response schemas (`models.py`)
- **Fallback** → switches from quantum (QKD) to classical (KMS) if needed
- **Orchestration** → manages complete handshake lifecycle inside one endpoint
- **Service Layer** (`main.py`) →
    - **/handshake [POST]** → single entrypoint that:
        - Calls **KMS Service** → create key (selected algorithm, TTL, material)
        - Calls **KDE Service** → deliver key material to destination
        - Calls **Crypto Modules** → perform encryption with selected key (produce ciphertext + nonce)
        - Calls **Send Validation API** → send encrypted validation package back to origin/receiver

---
### **key_destination_engine**
- **Delivery Orchestration** → manages secure key delivery to multiple destination types (`destination_engine.py`)
- **Multi-Transport Support** → supports API, MQTT, HSM, and FILE delivery methods (`config.py`)
- **Request Modeling** → structured delivery requests and responses (`models.py`)
- **Handler Registry** → pluggable delivery method handlers (api_delivery, mqtt_delivery, hsm_delivery, file_delivery)
- **Tracking & Status** → in-memory delivery tracking with attempt counting and status monitoring
- **Service Endpoints** (`main.py`) →
    - **/deliver [POST]** → accepts delivery request, routes to appropriate handler, returns delivery status
    - **/delivery/{id} [GET]** → retrieves specific delivery status by delivery_id
    - **/deliveries [GET]** → lists all tracked deliveries
    - **/health [GET]** → health check endpoint

---
### **crypto_module**
- **AEAD Encryption** → performs authenticated encryption using AES256_GCM and ChaCha20Poly1305 (`crypto_engine.py`)
- **Key Derivation** → uses HKDF to derive encryption/decryption keys from KMS key material
- **Multi-Source Integration** → fetches keys from **KMS Service** and messages from **Interceptor API**
- **Request Modeling** → structured encrypt/decrypt requests and responses (`models.py`)
- **Session Management** → handles session-based key context with expiration validation
- **Service Endpoints** (`main.py`) →
    - **/encrypt [POST]** → encrypts plaintext (inline or fetched from Interceptor); calls **KMS Service** for key context
    - **/encrypt/by-request-id [POST]** → simplified encrypt using request_id + session_id; calls **Interceptor API** for message and **KMS Service** for key
    - **/decrypt [POST]** → decrypts ciphertext using session key; calls **KMS Service** for key context
    - **/health [GET]** → health check endpoint

---
### **validation_send_api**
- **Payload Reception** → receives encrypted negotiation payloads from **Handshake Negotiator** (`NegotiationPayload`)
- **Persistence Layer** → stores encrypted delivery records in database with status tracking (`EncryptedDelivery`, `EncryptedDeliveryRepository`)
- **Forward Service** → forwards encrypted payloads to origin/receiver URLs via HTTP POST (`ForwardService`)
- **Status Management** → tracks delivery status (pending → delivered/failed) with error logging
- **Transaction Safety** → uses `@Transactional` for atomic persist-and-forward operations
- **Service Endpoints** (`SendController`) →
    - **/validation/send [POST]** → receives payload from **Handshake Negotiator**, persists to database, forwards to `originUrl`, updates status
        - On success → marks as "delivered", returns 200
        - On failure → marks as "failed" with error message, returns 502

## 1. **Interceptor API**

### `POST /intercept`

**Description:** Receives and stores an intercepted message with metadata.

**Request Body Parameters:**

- `message` _(string, required)_ → The actual message content intercepted.
- `sourceId` _(string, required)_ → Identifier of the source system/app.
- `destinationId` _(string, required)_ → Identifier of the destination system/app.
- `metadata` _(object, optional)_ → Arbitrary metadata (e.g. priority, headers).

**Example Request:**

json

Copy

{  
  "message": "Hello, world!",  
  "sourceId": "app1",  
  "destinationId": "app2",  
  "metadata": { "priority": "high" }  
}  

---

### `GET /intercept/message?request_id={id}`

**Description:** Retrieve the message previously intercepted using a request ID.

**Query Parameters:**

- `request_id` _(string, required)_ → The unique ID of the intercepted request.

---

### `GET /health`

**Description:** Service health check.

---

## 2. **Context API**

### `POST /context/enrich` (detailed enrichment)

**Description:** Builds a source/destination context using full details.

**Body Parameters:**

- `requestId` _(string, required)_ → Unique request identifier.
- `source` _(object, required)_ → Information about the source user/device:
    - `ip` _(string)_ → Source IP address.
    - `user_id` _(string)_ → User identifier.
    - `device_id` _(string)_ → Device identifier.
    - `user_agent` _(string)_ → User agent string.
    - `geo` _(string)_ → Geo-location or country code.
    - `os_version` _(string)_ → Operating system version.
    - `device_type` _(string)_ → Device type (mobile, laptop, server).
    - `mfa_status` _(string)_ → MFA state ("enabled"/"disabled").
    - `security_status` _(string)_ → Security health (e.g. "healthy").
- `destination` _(object, required)_ → Destination service info:
    - `ip`, `service_id`, `service_type`, `security_policy`, `security_status`, `os_version`, `allowed_protocols[]`.
- `metadata` _(object, optional)_ → Extra transaction metadata (`transaction_id`, `operation`).

---

### `POST /context/enrich` (hint mode)

Simplified version using _hints_ instead of full details.

- `source_hint` _(object)_ → Partial hints like IP, user agent.
- `destination_hint` _(object)_ → Destination hints (host, service_id, path).
- `headers` _(object)_ → Extra headers to enrich.
- `content_pointer` _(object)_ → Reference to content/text + MIME type.

---

### `POST /context/enrich/simple`

Minimal context creation.

- `sourceId` _(string)_
- `destinationId` _(string)_
- `content` _(string)_ → Message/text to contextualize.
- `metadata` _(object)_ → Additional info such as `host`, `path`, `user_agent`, `device_id`.

---

### `GET /context/record?request_id={id}`

Fetch an enriched context payload by request ID.

---

## 3. **Risk API**

### `POST /risk/train`

**Body Parameters:**

- `n` _(integer, required)_ → Number of samples to train on.
- `seed` _(integer, optional)_ → Randomization seed for reproducibility.

---

### `POST /risk/assess`

**Body Parameters:**

- `request_id` _(string, required)_ → Target request ID.
- `signals` _(object, required)_ → Risk signals set:
    - `global_alert_level` _(string)_ → low/medium/high.
    - `current_campaigns[]` _(array)_ → Ongoing threats (name, severity, geo).
    - `anomaly_index_global` _(float)_
    - `incident_rate_7d` _(integer)_
    - `patch_delay_days_p50` _(integer)_
    - `exposure_level` _(string)_
    - `maintenance_window` _(boolean)_
    - `compliance_debt_score` _(float)_
    - `business_critical_period` _(boolean)_
    - `geo_region` _(string)_

---

### `POST /risk/cleanup/all`

Cleanup trained models.

- `dry_run` _(boolean)_ → If true, shows what would be cleaned without executing.

---

## 4. **Confidentiality API**

### `POST /confidentiality/classify`

**Body Parameters:**

- `request_id` _(string)_ → The request identifier.
- `content_pointer` _(object)_ →
    - `ref` _(string)_ → Document reference.
    - `sample_text` _(string)_ → Snippet of text to classify.
    - `metadata` _(object)_ → Document metadata, including `doc_type`, `app`, `user_label`.
- `source` _(object)_ → Source info (IP).
- `destination` _(object)_ → Destination info (service_id).

---

### `POST /confidentiality/train`

Starts training the confidentiality models (form body).

---

### `POST /confidentiality/cleanup/all`

Cleans up existing models.

- `dry_run` _(bool)_

---

## 5. **Handshake Negotiator**

### `POST /handshake`

**Body Parameters:**

- `request_id` _(string, optional)_ → Custom ID or auto-generated.
- `source` _(string, required)_ → Origin node identifier.
- `destination` _(string, required)_ → Target node or URL.
- `dst_props` _(object, optional)_ → Destination properties:
    - `hardware[]` _(array)_ → Capabilities (QKD, PQC, CPU_ONLY, etc.).
    - `compliance[]` _(array)_ → Compliance needs (HIPAA, GDPR, PCI-DSS).
    - `max_latency_ms` _(integer)_ → Max acceptable latency.
- `proposed[]` _(array, required)_ → Proposed algorithms (QKD_BB84, Kyber1024, AES256_GCM, etc.).

---

## 6. **KMS Service**

### `POST /kms/create_key`

**Body Parameters:**

- `session_id` _(string, optional)_ → Preferred session ID (or auto-generated).
- `request_id` _(string, optional)_ → Associated request.
- `algorithm` _(string, required)_ → Algorithm for the key (AES256_GCM, Kyber, etc.).
- `ttl_seconds` _(integer, required)_ → Key validity in seconds.
- `strict` _(boolean, optional)_ → If true, strictly enforce requested algorithm.

---

### `GET /kms/get_key/{session_id}`

Retrieve specific key context by session ID.

### `GET /kms/get_key?request_id={id}`

Retrieve key context associated with a request.

---

### `GET /kms/supported_algorithms`

Get a list of supported algorithms.

---

## 7. **Key Destination Engine (KDE)**

### `POST /deliver`

**Body Parameters:**

- `session_id` _(string, required)_ → Session ID of the key.
- `request_id` _(string, required)_ → Request ID.
- `destination` _(string, required)_ → URL or identifier of destination service.
- `delivery_method` _(string, required)_ → One of ["API","MQTT","HSM","FILE"].
- `key_material` _(string, required)_ → Base64 key material.
- `algorithm` _(string, required)_ → Encryption algorithm.
- `expires_at` _(int or string, required)_ → Expiration (epoch or ISO8601).
- `metadata` _(object, optional)_ → Extra options (headers etc.).

---

### `GET /delivery/{id}`

Get status of a delivery.

### `GET /deliveries`

List all deliveries.

---

## 8. **Crypto Module**

### `POST /encrypt`

**Body Parameters:**

- `session_id` _(string, required)_ → Get key from KMS.
- `request_id` _(string, optional)_ → For message lookup in Interceptor.
- `algorithm` _(string, required)_ → "AES256_GCM" or "CHACHA20_POLY1305".
- `plaintext_b64` _(string, optional)_ → Plaintext in Base64 (inline).
- `fetch_from_interceptor` _(boolean, optional, default=true)_ → If true, fetch plaintext from Interceptor when not provided.
- `aad_b64` _(string, optional)_ → Additional Authenticated Data (AAD).

---

### `POST /encrypt/by-request-id`

**Body Parameters:**

- `request_id` _(string, required)_
- `session_id` _(string, required)_
- `algorithm` _(string, default=AES256_GCM)_
- `aad_b64` _(string, optional)_

---

### `POST /decrypt`

**Body Parameters:**

- `session_id` _(string, required)_ → Fetches KMS key.
- `request_id` _(string, optional)_
- `algorithm` _(string, required)_
- `nonce_b64` _(string, required)_
- `ciphertext_b64` _(string, required)_
- `aad_b64` _(string, optional)_

---

## 9. **Validation Send API**

### `POST /validation/send`

**Body Parameters:**

- `requestId` _(string, required)_ → Original request.
- `sessionId` _(string, required)_ → Session ID from KMS.
- `selectedAlgorithm` _(string, optional)_ → Agreed algorithm.
- `cryptoNonceB64` _(string, required)_ → Nonce from encryption.
- `cryptoCiphertextB64` _(string, required)_ → Ciphertext from encryption.
- `cryptoAlgorithm` _(string, required)_
- `cryptoExpiresAt` _(int, optional)_ → Expiry (epoch).
- `sourceId` _(string, optional)_ → Source entity ID.
- `originUrl` _(string, required)_ → Where to forward the payload.

---

## 10. **Receiver API**

### `POST /receiver`

**Body Parameters:**

- `request_id` _(string, required)_
- `session_id` _(string, required)_
- `key_material` _(string, base64, required)_
- `algorithm` _(string, required)_

---

### `POST /decrypt`

**Body Parameters:**

- `request_id` _(string, required)_
- `nonce_b64` _(string, required)_
- `ciphertext_b64` _(string, required)_
- `algorithm` _(string, required)_

---

### `GET /keys`

List available stored keys.

### `GET /logs?limit=N`

Retrieve last **N** log entries.