# Classification Agent - Technical API Specification

## Introduction
The Classification Agent is a critical component of the Q-OPSEC cluster, responsible for real-time traffic classification and machine learning inference. It provides endpoints for model management, data prediction, and dataset orchestration.

## Global Configuration
- **Base URL:** `http://192.168.18.18:8088`
- **Protocol:** HTTP/1.1
- **Auth:** X-API-Key required for administrative endpoints.

## Module: app/api/v1/endpoints.py

### GET /api/v1/health
**Summary:** Health check do serviço

**Description:**
No detailed description.

**Input Parameters:**
```python

```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/model
**Summary:** Informações do modelo carregado

**Description:**
No detailed description.

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/model/reload
**Summary:** Recarrega o modelo mais recente

**Description:**
Recarrega o modelo mais recente disponível.

- Se `force=false` (padrão): só recarrega se houver uma versão mais nova
- Se `force=true`: força o reload mesmo se o modelo já estiver atualizado

Requer autenticação.

**Input Parameters:**
```python
request: ModelReloadRequest = ModelReloadRequest(),
    user: Dict[str, Any] = Depends(require_auth),
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/model/manifest
**Summary:** Manifest do modelo (schema de entrada/saída)

**Description:**
Retorna o manifest completo do modelo, incluindo:

- Schema de entrada (features requeridas e tipos)
- Schema de saída (formato da predição)
- Classes disponíveis
- Metadados do modelo

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/predict
**Summary:** Executa predição/classificação

**Description:**
Executa predição usando o modelo carregado.

### Fluxo
1. **Validação**: Valida o input contra o schema do modelo
2. **Predição**: Executa a predição e retorna label + probabilidades
3. **RL Engine** (opcional): Se `send_to_rl=true`, envia resultado para RL Engine

### Input
- `data`: Objeto ou lista de objetos com as features
- `return_probabilities`: Se deve retornar probabilidades (padrão: true)
- `send_to_rl`: Se deve enviar para RL Engine (padrão: false)

### Output
- `results`: Lista de predições (label, confidence, probabilities)
- `model_name`, `model_version`: Informações do modelo usado
- `prediction_time_ms`: Tempo de processamento

**Input Parameters:**
```python
request: PredictionRequest,
    http_request: Request,
    user: Dict[str, Any] = Depends(get_current_user),
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/metrics
**Summary:** Métricas do serviço

**Description:**
Retorna métricas agregadas do serviço, incluindo:

- Total de requisições e predições
- Tempo médio de resposta
- Taxa de erro
- Número de reloads do modelo
- Uptime
- Timestamp da última predição

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(require_auth)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/sessions
**Summary:** Lista sessões de treinamento

**Description:**
No detailed description.

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/latest
**Summary:** Summary da sessão de treinamento mais recente

**Description:**
No detailed description.

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/{session_id}/summary
**Summary:** Summary de uma sessão específica

**Description:**
No detailed description.

**Input Parameters:**
```python
session_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/images
**Summary:** Lista imagens disponíveis

**Description:**
No detailed description.

**Input Parameters:**
```python
user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/{session_id}/images/{image_name}
**Summary:** Download de imagem de uma sessão

**Description:**
No detailed description.

**Input Parameters:**
```python
session_id: str,
    image_name: str,
    user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/training/latest/images/{image_name}
**Summary:** Download de imagem da sessão mais recente

**Description:**
No detailed description.

**Input Parameters:**
```python
image_name: str,
    user: Dict[str, Any] = Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

## Module: app/api/v1/endpoints_datasets.py

### GET /api/v1/datasets
**Summary:** Lista todos os datasets

**Description:**
Lista todos os datasets disponíveis no diretório configurado.

Para cada dataset, retorna:
- Nome
- Path completo
- Tamanho total (soma de todos os arquivos)
- Lista de arquivos
- Timestamp da última modificação

**Input Parameters:**
```python
user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/datasets
**Summary:** Cria um novo dataset

**Description:**
Cria um novo dataset (diretório vazio).

O nome do dataset será usado como nome do diretório.
Retorna erro 409 se o dataset já existir.

Requer autenticação.

**Input Parameters:**
```python
payload: CreateDatasetRequest, user=Depends(require_auth)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}
**Summary:** Informações de um dataset específico

**Description:**
No detailed description.

**Input Parameters:**
```python
name: str, user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### DELETE /api/v1/datasets/{name}
**Summary:** Deleta um dataset

**Description:**
Deleta um dataset e todos os seus arquivos.

**ATENÇÃO**: Esta operação é irreversível!

Requer autenticação.

**Input Parameters:**
```python
name: str, user=Depends(require_auth)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/datasets/{name}/files
**Summary:** Upload de arquivo para um dataset

**Description:**
Faz upload de um arquivo para um dataset.

- Se o arquivo já existir, use `override=true` para sobrescrever
- O arquivo é salvo com o nome original (`file.filename`)

Requer autenticação.

**Input Parameters:**
```python
name: str,
        file: UploadFile = File(...),
        override: bool = Query(False, description="Sobrescrever se o arquivo já existir"),
        user=Depends(require_auth),
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/files
**Summary:** Lista arquivos de um dataset

**Description:**
No detailed description.

**Input Parameters:**
```python
name: str, user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/files/{filename}
**Summary:** Download de arquivo

**Description:**
No detailed description.

**Input Parameters:**
```python
name: str, filename: str, user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### DELETE /api/v1/datasets/{name}/files/{filename}
**Summary:** Deleta um arquivo

**Description:**
No detailed description.

**Input Parameters:**
```python
name: str, filename: str, user=Depends(require_auth)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/datasets/{name}/metadata
**Summary:** Define metadados do dataset

**Description:**
Define ou atualiza os metadados de um dataset.

Os metadados são salvos em `metadata.json` no diretório do dataset.

Requer autenticação.

**Input Parameters:**
```python
name: str, metadata: Dict[str, Any], user=Depends(require_auth)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/metadata
**Summary:** Obtém metadados do dataset

**Description:**
No detailed description.

**Input Parameters:**
```python
name: str, user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/preview
**Summary:** Preview de dados

**Description:**
Retorna um preview (primeiras N linhas) de um arquivo tabular.

Suporta:
- CSV, TSV
- Parquet
- JSONL, NDJSON

Parâmetros:
- `file`: Nome do arquivo
- `n`: Número de linhas (padrão: 50, máx: 500)

**Input Parameters:**
```python
name: str,
        file: str = Query(..., description="Nome do arquivo"),
        n: int = Query(50, ge=1, le=500, description="Número de linhas"),
        user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/schema
**Summary:** Inferência de schema

**Description:**
Infere o schema de um arquivo tabular.

Retorna:
- Lista de colunas
- Tipos de dados (dtypes)
- Shape (linhas, colunas)
- Contagem de valores nulos por coluna

**Input Parameters:**
```python
name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### GET /api/v1/datasets/{name}/stats
**Summary:** Estatísticas descritivas

**Description:**
Retorna estatísticas descritivas de um arquivo tabular.

Para colunas numéricas:
- Estatísticas descritivas (mean, std, min, max, quartis)

Para colunas categóricas:
- Top 10 valores mais frequentes

**Input Parameters:**
```python
name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

### POST /api/v1/datasets/{name}/validate
**Summary:** Valida dataset contra modelo

**Description:**
Valida se um arquivo de dataset é compatível com o modelo carregado.

Verifica:
- Se todas as colunas requeridas pelo modelo estão presentes
- Quais colunas estão faltando
- Quais colunas extras existem (não requeridas)

Usa uma amostra de 200 linhas para validação.

**Input Parameters:**
```python
name: str,
        file: str = Query(..., description="Nome do arquivo"),
        user=Depends(get_current_user)
```

**Response:**
- `200 OK`: Successful operation.
- `401 Unauthorized`: Missing or invalid API Key.
- `404 Not Found`: Target resource (model/dataset) does not exist.

---

