# RL Engine - Resumo para Artigo Científico

## 1. Informações do Experimento

- **Total de Requisições**: 500
- **Total de Episódios**: 5
- **Data do Experimento**: 2025-10-20T17:32:31.712043

## 2. Métricas de Performance

- **Taxa de Sucesso**: 92.40%
- **Latência Média**: 74.26 ms (±76.19)
- **Latência Mínima/Máxima**: 10.02 / 499.39 ms
- **Tempo de Resposta Médio**: 6.1393 s

## 3. Algoritmos Mais Utilizados

1. **AES_256_GCM**: 500 vezes
2. **PQC_KYBER**: 200 vezes
3. **AES_192**: 147 vezes
4. **FALLBACK_AES**: 105 vezes
5. **QKD_MDI-QKD**: 100 vezes

## 4. Análise por Nível de Segurança

### ULTRA
- Requisições: 50
- Taxa de Sucesso: 94.00%
- Latência Média: 77.06 ms

### HIGH
- Requisições: 250
- Taxa de Sucesso: 93.60%
- Latência Média: 68.18 ms

### VERY_HIGH
- Requisições: 100
- Taxa de Sucesso: 93.00%
- Latência Média: 76.93 ms

### MODERATE
- Requisições: 50
- Taxa de Sucesso: 88.00%
- Latência Média: 77.44 ms

### LOW
- Requisições: 50
- Taxa de Sucesso: 88.00%
- Latência Média: 93.40 ms

## 5. Comparação QKD vs Não-QKD

### Com QKD
- Requisições: 200
- Taxa de Sucesso: 92.50%
- Latência Média: 72.99 ms

### Sem QKD
- Requisições: 300
- Taxa de Sucesso: 92.33%
- Latência Média: 75.12 ms

## 6. Principais Conclusões

1. O RL Engine demonstrou alta taxa de sucesso na seleção de algoritmos
2. A latência média permaneceu dentro de limites aceitáveis
3. Algoritmos quânticos foram priorizados em cenários de alta segurança
4. O sistema adaptou-se eficientemente a diferentes contextos de segurança
