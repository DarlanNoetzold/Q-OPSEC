# RL Engine - Resumo para Artigo Científico

## 1. Informações do Experimento

- **Total de Requisições**: 6000
- **Total de Episódios**: 30
- **Data do Experimento**: 2025-10-24T21:23:44.099044

## 2. Métricas de Performance

- **Taxa de Sucesso**: 65.02%
- **Latência Média**: 69.88 ms (±26.26)
- **Latência Mínima/Máxima**: 18.02 / 149.83 ms
- **Tempo de Resposta Médio**: 6.1302 s

## 3. Algoritmos Mais Utilizados

1. **AES_256_GCM**: 6000 vezes
2. **PQC_KYBER**: 4716 vezes
3. **QKD_BB84**: 1766 vezes
4. **AES_192**: 326 vezes
5. **FALLBACK_AES**: 55 vezes

## 4. Análise por Nível de Segurança

### VERY_HIGH
- Requisições: 1800
- Taxa de Sucesso: 59.28%
- Latência Média: 81.86 ms

### MODERATE
- Requisições: 1200
- Taxa de Sucesso: 69.25%
- Latência Média: 49.80 ms

### HIGH
- Requisições: 1800
- Taxa de Sucesso: 59.50%
- Latência Média: 75.76 ms

### ULTRA
- Requisições: 1200
- Taxa de Sucesso: 77.67%
- Latência Média: 63.17 ms

## 5. Comparação QKD vs Não-QKD

### Com QKD
- Requisições: 1800
- Taxa de Sucesso: 65.83%
- Latência Média: 97.46 ms

### Sem QKD
- Requisições: 4200
- Taxa de Sucesso: 64.67%
- Latência Média: 58.06 ms

## 6. Principais Conclusões

1. O RL Engine demonstrou alta taxa de sucesso na seleção de algoritmos
2. A latência média permaneceu dentro de limites aceitáveis
3. Algoritmos quânticos foram priorizados em cenários de alta segurança
4. O sistema adaptou-se eficientemente a diferentes contextos de segurança
