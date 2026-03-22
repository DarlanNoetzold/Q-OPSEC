#!/bin/bash
# ============================================================
# Q-OPSEC KMS v3.0 - Standalone Test Script
# Testa todos os endpoints do KMS isoladamente
# Requisito: KMS rodando em http://localhost:8002
# ============================================================

KMS_URL="http://localhost:8002"
PASS=0
FAIL=0

green() { echo -e "\033[32m$1\033[0m"; }
red() { echo -e "\033[31m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }

check() {
    local test_name="$1"
    local response="$2"
    local expected="$3"
    if echo "$response" | grep -q "$expected"; then
        green "  [PASS] $test_name"
        PASS=$((PASS+1))
    else
        red "  [FAIL] $test_name"
        echo "    Expected: $expected"
        echo "    Got: $response"
        FAIL=$((FAIL+1))
    fi
}

echo "============================================================"
echo "  Q-OPSEC KMS v3.0 - Standalone Tests"
echo "============================================================"
echo ""

# --------------------------------------------------
yellow "1. Health Check"
# --------------------------------------------------
RESP=$(curl -s $KMS_URL/health)
check "Health endpoint returns 'healthy'" "$RESP" '"status":"healthy"'
check "Health shows version 3.0.0" "$RESP" '"version":"3.0.0"'
check "Health reports qkd_gateway status" "$RESP" '"qkd_gateway"'
check "Health reports storage backend" "$RESP" '"backend":"memory"'

echo ""

# --------------------------------------------------
yellow "2. Supported Algorithms"
# --------------------------------------------------
RESP=$(curl -s $KMS_URL/kms/supported_algorithms)
check "Returns classical algorithms" "$RESP" '"AES256_GCM"'
check "Returns PQC KEMs (Kyber)" "$RESP" '"Kyber768"'
check "Returns PQC KEMs (NTRU)" "$RESP" '"NTRU-HRSS-701"'
check "Returns PQC KEMs (FrodoKEM)" "$RESP" '"FrodoKEM-640-AES"'
check "Returns PQC KEMs (McEliece)" "$RESP" '"Classic-McEliece-348864"'
check "Returns PQC KEMs (HQC)" "$RESP" '"HQC-128"'
check "Returns PQC KEMs (BIKE)" "$RESP" '"BIKE-L1"'
check "Returns PQC KEMs (Saber)" "$RESP" '"Saber"'
check "Returns PQC Signatures" "$RESP" '"Dilithium2"'
check "Returns QKD algorithms" "$RESP" '"QKD_BB84"'

echo ""

# --------------------------------------------------
yellow "3. Algorithm Info"
# --------------------------------------------------
RESP=$(curl -s $KMS_URL/kms/algorithm_info/AES256_GCM)
check "AES256_GCM info returns symmetric" "$RESP" '"category":"symmetric"'

RESP=$(curl -s $KMS_URL/kms/algorithm_info/Kyber768)
check "Kyber768 info returns pqc_kem" "$RESP" '"category":"pqc_kem"'
check "Kyber768 is quantum_resistant" "$RESP" '"quantum_resistant":true'

RESP=$(curl -s $KMS_URL/kms/algorithm_info/QKD_BB84)
check "QKD_BB84 info returns qkd" "$RESP" '"category":"qkd"'

RESP=$(curl -s -o /dev/null -w "%{http_code}" $KMS_URL/kms/algorithm_info/INVALID_ALGO)
check "Unknown algorithm returns 404" "$RESP" "404"

echo ""

# --------------------------------------------------
yellow "4. Create Key - Classical Algorithms"
# --------------------------------------------------
for ALGO in AES256_GCM AES128_GCM ChaCha20_Poly1305 RSA2048 ECDH_P256; do
    RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
      -H "Content-Type: application/json" \
      -d "{\"request_id\": \"req_classic_${ALGO}\", \"algorithm\": \"${ALGO}\", \"ttl_seconds\": 300}")
    check "Create key $ALGO" "$RESP" '"source_of_key":"classical"'
done

echo ""

# --------------------------------------------------
yellow "5. Create Key - PQC Algorithms"
# --------------------------------------------------
for ALGO in Kyber512 Kyber768 Kyber1024 Dilithium2 NTRU-HRSS-701 FrodoKEM-640-AES HQC-128 BIKE-L1 Saber Classic-McEliece-348864; do
    RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
      -H "Content-Type: application/json" \
      -d "{\"request_id\": \"req_pqc_${ALGO}\", \"algorithm\": \"${ALGO}\", \"ttl_seconds\": 300}")
    check "Create key $ALGO" "$RESP" '"key_material"'
done

echo ""

# --------------------------------------------------
yellow "6. Create Key - QKD Algorithms"
# --------------------------------------------------
RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_qkd_bb84", "algorithm": "QKD_BB84", "ttl_seconds": 300}')
check "Create key QKD_BB84 returns key" "$RESP" '"key_material"'
check "QKD_BB84 source is qkd" "$RESP" '"source_of_key":"qkd"'
check "QKD_BB84 has qkd_metadata" "$RESP" '"qkd_metadata"'
check "QKD_BB84 has qber" "$RESP" '"qber"'

echo ""

# --------------------------------------------------
yellow "7. Key Retrieval by session_id"
# --------------------------------------------------
RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_retrieve_sess", "algorithm": "AES256_GCM", "ttl_seconds": 600}')
SESSION_ID=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
KEY_MATERIAL=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['key_material'])")

RESP=$(curl -s $KMS_URL/kms/get_key/$SESSION_ID)
check "Get key by session_id returns correct session" "$RESP" "\"session_id\":\"$SESSION_ID\""
check "Get key by session_id returns correct key_material" "$RESP" "\"key_material\":\"$KEY_MATERIAL\""

echo ""

# --------------------------------------------------
yellow "8. Key Retrieval by request_id"
# --------------------------------------------------
RESP=$(curl -s "$KMS_URL/kms/get_key?request_id=req_retrieve_sess")
check "Get key by request_id returns session" "$RESP" "\"session_id\":\"$SESSION_ID\""
check "Get key by request_id returns key_material" "$RESP" "\"key_material\":\"$KEY_MATERIAL\""

echo ""

# --------------------------------------------------
yellow "9. Session Delete"
# --------------------------------------------------
RESP=$(curl -s -X DELETE $KMS_URL/kms/session/$SESSION_ID)
check "Delete session returns success" "$RESP" '"message":"Session deleted successfully"'

RESP=$(curl -s -o /dev/null -w "%{http_code}" $KMS_URL/kms/get_key/$SESSION_ID)
check "Deleted session returns 404" "$RESP" "404"

echo ""

# --------------------------------------------------
yellow "10. Fallback Behavior"
# --------------------------------------------------
RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_fallback_test", "algorithm": "UNKNOWN_ALGO_XYZ", "ttl_seconds": 300}')
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"request_id": "req_fallback_test2", "algorithm": "UNKNOWN_ALGO_XYZ", "ttl_seconds": 300}')
check "Unknown algorithm returns 400" "$HTTP_CODE" "400"

echo ""

# --------------------------------------------------
yellow "11. Hardware Profile"
# --------------------------------------------------
RESP=$(curl -s $KMS_URL/kms/hardware_profile)
check "Hardware profile returns capability_tier" "$RESP" '"capability_tier"'
check "Hardware profile returns cpu info" "$RESP" '"architecture"'
check "Hardware profile returns memory info" "$RESP" '"total_ram_gb"'
check "Hardware profile returns aes_ni status" "$RESP" '"aes_ni"'

echo ""

# --------------------------------------------------
yellow "12. Input Validation"
# --------------------------------------------------
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "AES256_GCM", "ttl_seconds": 300}')
check "Missing both session_id and request_id returns 422" "$HTTP_CODE" "422"

echo ""

# --------------------------------------------------
yellow "13. Custom session_id"
# --------------------------------------------------
RESP=$(curl -s -X POST $KMS_URL/kms/create_key \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_custom_session_123", "request_id": "req_custom_id", "algorithm": "AES256_GCM", "ttl_seconds": 300}')
check "Custom session_id is preserved" "$RESP" '"session_id":"my_custom_session_123"'

echo ""

# ==================================================
echo "============================================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================================"

if [ $FAIL -eq 0 ]; then
    green "  ALL TESTS PASSED"
else
    red "  SOME TESTS FAILED"
fi
