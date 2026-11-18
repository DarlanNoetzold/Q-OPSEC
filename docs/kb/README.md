# Base de conhecimento — Q-OPSEC

Índice da base de conhecimento do projeto. Cada arquivo cobre um tema e é escrito para ser lido
antes de mexer na área correspondente.

| Arquivo | Conteúdo |
| --- | --- |
| [00-pendencias.md](00-pendencias.md) | Backlog técnico aberto, com evidência, critério de aceite e comando de verificação |
| [01-escala-horizontal.md](01-escala-horizontal.md) | O que já foi implementado para escala (KMS stateless, fila, resiliência, rate limit, deploy) e como validar |

## Como usar

1. Antes de começar uma tarefa, procure o item em `00-pendencias.md` pelo ID (`KB-00x`).
2. Ao concluir, mova o item para a seção "Concluído" com a data e a evidência da verificação.
3. Não marque como concluído sem ter rodado o comando de verificação descrito no item.

## Convenções

- **ID**: `KB-NNN`, estável, nunca reutilizado.
- **Status**: `aberto`, `em andamento`, `bloqueado`, `concluído`.
- **Evidência**: saída de comando, teste ou arquivo que prova o estado atual.
- **Aceite**: condição objetiva que fecha o item.
