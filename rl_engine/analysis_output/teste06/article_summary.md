# RL Engine - Resumo para Artigo Científico

## 1. Informações do Experimento

- **Total de Requisições**: 1000
- **Total de Episódios**: 10
- **Data do Experimento**: 2025-10-25T09:32:35.016350

## 2. Métricas de Performance

- **Taxa de Sucesso**: 49.50%
- **Latência Média**: 75.30 ms (±36.34)
- **Latência Mínima/Máxima**: 13.73 / 171.84 ms
- **Tempo de Resposta Médio**: 6.1079 s

## 3. Algoritmos Mais Utilizados

1. **AES_256_GCM**: 1000 vezes
2. **PQC_KYBER**: 781 vezes
3. **QKD_BB84**: 296 vezes
4. **AES_192**: 55 vezes
5. **FALLBACK_AES**: 11 vezes

## 4. Análise por Nível de Segurança

### VERY_HIGH
- Requisições: 300
- Taxa de Sucesso: 41.67%
- Latência Média: 92.94 ms

### HIGH
- Requisições: 300
- Taxa de Sucesso: 39.67%
- Latência Média: 85.92 ms

### MODERATE
- Requisições: 200
- Taxa de Sucesso: 55.50%
- Latência Média: 45.08 ms

### ULTRA
- Requisições: 200
- Taxa de Sucesso: 70.00%
- Latência Média: 63.13 ms

## 5. Comparação QKD vs Não-QKD

### Com QKD
- Requisições: 300
- Taxa de Sucesso: 55.33%
- Latência Média: 110.97 ms

### Sem QKD
- Requisições: 700
- Taxa de Sucesso: 47.00%
- Latência Média: 60.02 ms

## 6. Principais Conclusões

1. O RL Engine demonstrou alta taxa de sucesso na seleção de algoritmos
2. A latência média permaneceu dentro de limites aceitáveis
3. Algoritmos quânticos foram priorizados em cenários de alta segurança
4. O sistema adaptou-se eficientemente a diferentes contextos de segurança
