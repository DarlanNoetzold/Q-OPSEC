#!/bin/bash
# ============================================================
# Q-OPSEC Middleware - Full Flow Test Script
#
# Pré-requisitos (todos os serviços rodando):
#   - Interceptor API     : http://localhost:8080
#   - Context API         : http://localhost:65534
#   - Risk Service        : http://localhost:8000
#   - Confidentiality     : http://localhost:8083
#   - Classification Agent: http://localhost:8088
#   - RL Engine           : http://localhost:9009
#   - Handshake Negotiator: http://localhost:8001
#   - KMS Service         : http://localhost:8002
#   - KDE                 : http://localhost:8003
#   - Crypto Module       : http://localhost:8004
#   - Validation Send API : http://localhost:8005
# ============================================================

KMS_URL="http://localhost:8002"
HANDSHAKE_URL="http://localhost:8001"
CRYPTO_URL="http://localhost:8004"
KDE_URL="http://localhost:8003"
INTERCEPTOR_URL="http://localhost:8080"
CONTEXT_URL="http://localhost:65534"
RL_URL="http://localhost:9009"
CLASSIFY_URL="http://localhost:8088"
VALIDATION_URL="http://localhost:8005"

green() { echo -e "\033[32m$1\033[0m"; }
red() { echo -e "\033[31m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }
cyan() { echo -e "\033[36m$1\033[0m"; }

echo "============================================================"
echo "  Q-OPSEC Middleware - Full Flow Test"
echo "============================================================"
echo ""

# ==================================================
cyan "STEP 0: Health Check de todos os servicos"
# ==================================================
echo ""
for svc in \
    "KMS:$KMS_URL/health" \
    "Handshake:$HANDSHAKE_URL/docs" \
    "Crypto:$CRYPTO_URL/health" \
    "KDE:$KDE_URL/health"; do
    NAME=$(echo $svc | cut -d: -f1)
    URL=$(echo $svc | cut -d: -f2-)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 $URL 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ]; then
        green "  [OK] $NAME ($URL)"
    else
        red "  [DOWN] $NAME ($URL) -> HTTP $HTTP_CODE"
    fi
done
echo ""

# ==================================================
cyan "STEP 1: Interceptar mensagem"
# ==================================================
echo ""
yellow "curl -X POST $INTERCEPTOR_URL/intercept"
cat << 'EOF'
{
  "message": "Dados sensíveis: relatório financeiro Q1 2026",
  "sourceId": "app_financeiro",
  "destinationId": "app_auditoria",
  "metadata": {
    "priority": "high",
    "doc_type": "financial_report",
    "classification": "confidential"
  }
}
EOF
echo ""
echo "# Salvar o request_id retornado para os próximos passos"
echo ""

# ==================================================
cyan "STEP 2: Enriquecer contexto"
# ==================================================
echo ""
yellow "curl -X POST $CONTEXT_URL/context/enrich/simple"
cat << 'EOF'
{
  "sourceId": "app_financeiro",
  "destinationId": "app_auditoria",
  "content": "Dados sensíveis: relatório financeiro Q1 2026",
  "metadata": {
    "host": "server-fin-01",
    "path": "/api/reports",
    "user_agent": "Q-OPSEC-Client/1.0",
    "device_id": "dev_001"
  }
}
EOF
echo ""

# ==================================================
cyan "STEP 3: RL Engine decide algoritmos"
# ==================================================
echo ""
yellow "curl -X POST $RL_URL/act"
cat << 'EOF'
{
  "request_id": "req_<REQUEST_ID>",
  "source": "app_financeiro",
  "destination": "app_auditoria",
  "security_level": "Critical",
  "risk_score": 0.85,
  "conf_score": 0.90,
  "dst_props": {
    "hardware": ["PQC", "CPU_ONLY"],
    "compliance": ["LGPD", "PCI-DSS"],
    "max_latency_ms": 50
  }
}
EOF
echo ""
echo "# O RL Engine retorna proposed[] com os algoritmos sugeridos"
echo "# Ex: [\"Kyber1024\", \"AES256_GCM\"]"
echo ""

# ==================================================
cyan "STEP 4: Handshake Negotiator (chama KMS internamente)"
# ==================================================
echo ""
yellow "curl -X POST $HANDSHAKE_URL/handshake"
cat << 'EOF'
{
  "request_id": "req_<REQUEST_ID>",
  "source": "app_financeiro",
  "destination": "app_auditoria",
  "dst_props": {
    "hardware": ["PQC"],
    "compliance": ["LGPD"],
    "max_latency_ms": 50
  },
  "proposed": ["Kyber1024", "AES256_GCM"]
}
EOF
echo ""
echo "# O Handshake Negotiator internamente faz:"
echo "#   1. POST $KMS_URL/kms/create_key (cria chave com algoritmo negociado)"
echo "#   2. POST $KDE_URL/deliver (entrega chave ao destino)"
echo "#   3. POST $CRYPTO_URL/encrypt (cifra a mensagem)"
echo "#   4. POST $VALIDATION_URL/validation/send (envia pacote validado)"
echo ""

# ==================================================
cyan "STEP 5: KMS - Criar chave diretamente (teste isolado)"
# ==================================================
echo ""
yellow "# 5a. Criar chave AES256_GCM"
echo 'curl -s -X POST '$KMS_URL'/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id": "req_flow_001", "algorithm": "AES256_GCM", "ttl_seconds": 3600}'"'"
echo ""
echo "# Resposta esperada:"
echo '# { "session_id": "sess_...", "request_id": "req_flow_001", "selected_algorithm": "AES256_GCM",'
echo '#   "key_material": "base64...", "source_of_key": "classical", "fallback_applied": false }'
echo ""

yellow "# 5b. Criar chave PQC (Kyber768)"
echo 'curl -s -X POST '$KMS_URL'/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id": "req_flow_002", "algorithm": "Kyber768", "ttl_seconds": 3600}'"'"
echo ""

yellow "# 5c. Criar chave QKD (BB84)"
echo 'curl -s -X POST '$KMS_URL'/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id": "req_flow_003", "algorithm": "QKD_BB84", "ttl_seconds": 1800}'"'"
echo ""
echo "# Resposta inclui qkd_metadata com QBER, sifted_key_length, etc."
echo ""

# ==================================================
cyan "STEP 6: KMS - Recuperar chave (como o crypto_module faz)"
# ==================================================
echo ""
yellow "# 6a. Por session_id"
echo 'curl -s '$KMS_URL'/kms/get_key/{SESSION_ID}'
echo ""
yellow "# 6b. Por request_id"
echo 'curl -s "'$KMS_URL'/kms/get_key?request_id=req_flow_001"'
echo ""
echo "# Resposta: { session_id, request_id, algorithm, key_material, expires_at }"
echo ""

# ==================================================
cyan "STEP 7: Crypto Module - Encrypt (usa chave do KMS)"
# ==================================================
echo ""
yellow "curl -X POST $CRYPTO_URL/encrypt"
cat << 'EOF'
{
  "session_id": "<SESSION_ID do step 5>",
  "request_id": "req_flow_001",
  "algorithm": "AES256_GCM",
  "plaintext_b64": "RGFkb3Mgc2Vuc8OtdmVpczogcmVsYXTDs3JpbyBmaW5hbmNlaXJvIFExIDIwMjY=",
  "fetch_from_interceptor": false
}
EOF
echo ""
echo "# Retorna: { ciphertext_b64, nonce_b64, algorithm, session_id }"
echo ""

# ==================================================
cyan "STEP 8: Key Destination Engine - Entregar chave"
# ==================================================
echo ""
yellow "curl -X POST $KDE_URL/deliver"
cat << 'EOF'
{
  "session_id": "<SESSION_ID>",
  "request_id": "req_flow_001",
  "destination": "http://app_auditoria:9090/receiver",
  "delivery_method": "API",
  "key_material": "<KEY_MATERIAL base64>",
  "algorithm": "AES256_GCM",
  "expires_at": 1774200000,
  "metadata": {}
}
EOF
echo ""

# ==================================================
cyan "STEP 9: Validation Send - Enviar pacote cifrado"
# ==================================================
echo ""
yellow "curl -X POST $VALIDATION_URL/validation/send"
cat << 'EOF'
{
  "requestId": "req_flow_001",
  "sessionId": "<SESSION_ID>",
  "selectedAlgorithm": "AES256_GCM",
  "cryptoNonceB64": "<NONCE do step 7>",
  "cryptoCiphertextB64": "<CIPHERTEXT do step 7>",
  "cryptoAlgorithm": "AES256_GCM",
  "cryptoExpiresAt": 1774200000,
  "sourceId": "app_financeiro",
  "originUrl": "http://app_auditoria:9090/receiver"
}
EOF
echo ""

# ==================================================
cyan "STEP 10: Crypto Module - Decrypt (no receptor)"
# ==================================================
echo ""
yellow "curl -X POST $CRYPTO_URL/decrypt"
cat << 'EOF'
{
  "session_id": "<SESSION_ID>",
  "request_id": "req_flow_001",
  "algorithm": "AES256_GCM",
  "nonce_b64": "<NONCE do step 7>",
  "ciphertext_b64": "<CIPHERTEXT do step 7>"
}
EOF
echo ""

echo "============================================================"
echo ""
echo "============================================================"
cyan "  CURLS EXECUTÁVEIS PARA TESTE RÁPIDO DO KMS"
echo "============================================================"
echo ""

echo "# --- 1. Health ---"
echo 'curl -s http://localhost:8002/health | python3 -m json.tool'
echo ""

echo "# --- 2. Listar algoritmos ---"
echo 'curl -s http://localhost:8002/kms/supported_algorithms | python3 -m json.tool'
echo ""

echo "# --- 3. Info de algoritmo ---"
echo 'curl -s http://localhost:8002/kms/algorithm_info/Kyber768 | python3 -m json.tool'
echo 'curl -s http://localhost:8002/kms/algorithm_info/QKD_BB84 | python3 -m json.tool'
echo 'curl -s http://localhost:8002/kms/algorithm_info/AES256_GCM | python3 -m json.tool'
echo ""

echo "# --- 4. Criar chave AES ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_001","algorithm":"AES256_GCM","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 5. Criar chave Kyber768 ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_002","algorithm":"Kyber768","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 6. Criar chave NTRU ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_003","algorithm":"NTRU-HRSS-701","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 7. Criar chave FrodoKEM ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_004","algorithm":"FrodoKEM-640-AES","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 8. Criar chave HQC ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_005","algorithm":"HQC-128","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 9. Criar chave BIKE ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_006","algorithm":"BIKE-L1","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 10. Criar chave Classic McEliece ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_007","algorithm":"Classic-McEliece-348864","ttl_seconds":3600}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 11. Criar chave QKD BB84 ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_008","algorithm":"QKD_BB84","ttl_seconds":1800}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 12. Criar chave QKD E91 (demora ~2s por causa da simulacao) ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_009","algorithm":"QKD_E91","ttl_seconds":1800}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 13. Criar chave QKD MDI ---"
echo 'curl -s -X POST http://localhost:8002/kms/create_key \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"request_id":"req_010","algorithm":"QKD_MDI","ttl_seconds":1800}'"'"' | python3 -m json.tool'
echo ""

echo "# --- 14. Recuperar por session_id ---"
echo 'curl -s http://localhost:8002/kms/get_key/<SESSION_ID> | python3 -m json.tool'
echo ""

echo "# --- 15. Recuperar por request_id ---"
echo 'curl -s "http://localhost:8002/kms/get_key?request_id=req_001" | python3 -m json.tool'
echo ""

echo "# --- 16. Deletar sessao ---"
echo 'curl -s -X DELETE http://localhost:8002/kms/session/<SESSION_ID> | python3 -m json.tool'
echo ""

echo "# --- 17. Hardware Profile ---"
echo 'curl -s http://localhost:8002/kms/hardware_profile | python3 -m json.tool'
echo ""

echo "# --- 18. Benchmark (POST, pode demorar) ---"
echo 'curl -s -X POST http://localhost:8002/kms/benchmark \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'["AES256_GCM","Kyber512","Kyber768"]'"'"' | python3 -m json.tool'
echo ""
