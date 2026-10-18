# Teste dos modelos Agnes para o Videolyzer

Data: 2026-10-18 · Testado contra `https://apihub.agnes-ai.com/v1`
Vídeo de teste: 12s sintético com ground truth conhecido (0-4s: quadrado vermelho esq→dir; 4-8s: quadrado azul quicando vertical; 8-12s: quadrado verde estático)

## Descoberta principal

**Os "modelos de vídeo" da Agnes (`agnes-video-2.5-flash`, `agnes-video-2.5`) são de GERAÇÃO, não de análise.**
Evidências:
- `POST /v1/chat/completions` com esses modelos retorna: "Model agnes-video-2.5-flash is a video model. Use /v1/videos."
- `POST /v1/videos` exige `model` + `prompt` + `mode` (text-to-video e similares); campos `video` e `video_url` são rejeitados ("forbidden"/"not allowed")
- A doc oficial confirma: endpoint `/videos` é para geração, com polling assíncrono via `video_id`; cobrança é por "output resolution, output duration"
- Ou seja: **não servem para entender/analisar vídeos no pipeline do Videolyzer**

## Modelos de imagem

`agnes-image-2.0-flash`, `agnes-image-2.1-flash`, `agnes-image-2.5-flash` usam `/images/generations` (geração/edição de imagem, cobrança por resolução de saída). Também são geração, não análise. Não testados em chamada porque não agregam ao pipeline de análise.

## O que FUNCIONA: visão nos modelos de chat

Os modelos de chat aceitam `image_url` (base64) no `/v1/chat/completions` e enxergam bem:

| Modelo | Teste | Resultado | Latência |
|---|---|---|---|
| `agnes-2.5-flash` | frame t=2s (vermelho, esq) | Correto: "red square positioned on the far left side" | ~7.6s |
| `agnes-3.0-flash` | frame t=2s (vermelho, esq) | Correto: "red rectangle positioned on the left edge" | ~1.3s |
| `agnes-3.0-flash` | frame t=6s (azul, centro-topo) | Correto: "blue rectangle located at the top center" | ~4.5s |
| `agnes-3.0-flash` | 2 frames (t=2s + t=6s) com pergunta temporal | Excelente: descreveu o desaparecimento do vermelho e surgimento do azul com posições | ~5.5s |

O `agnes-3.0-flash` é **5-6x mais rápido** que o 2.5 e acertou tudo, inclusive comparação temporal entre 2 frames.

## Tabela comparativa (para o pipeline)

| Capacidade | Gemini 3.1 flash lite (atual) | Agnes video models | Agnes chat vision (`agnes-3.0-flash`) |
|---|---|---|---|
| Entendimento nativo de vídeo | Sim (Files API / file_data) | Não (só geram vídeo) | Não (só frames) |
| Análise frame-por-frame | Sim (via keyframes no modo OpenAI) | Não | **Sim, rápido (~1-5s/frame)** |
| Comparação temporal entre frames | Limitada | Não | **Sim, testado e correto** |
| Latência típica | ~40s (vídeo completo) | N/A | ~1-5s por chamada |
| Endpoint | generativelanguage.googleapis.com | /v1/videos (geração) | /v1/chat/completions |

## Recomendação

**Não vale** adicionar os modelos de vídeo da Agnes ao Videolyzer — são geradores, não analisadores. Seria como colocar um motor de carroça num avião.

**Vale** adicionar o `agnes-3.0-flash` como camada de análise frame-por-frame:
1. **Segunda opinião / fallback**: se o Gemini falhar ou demorar, rodar keyframes pelo Agnes
2. **Camada de verificação temporal**: o teste mostrou que ele compara 2 frames corretamente — dá para usar como verificador das cenas que o Gemini descreve ("o Gemini disse que X aconteceu entre 0:04-0:08, confirme com 2 frames")
3. **Modo Rápido**: para vídeos onde só interessa "o que aparece" (sem movimento fino), keyframes via modelo auxiliar saem em segundos contra ~40s do Gemini

Sugestão de implementação no Videolyzer: `--provider agnes` reaproveitando o padrão OpenAI-compatible já existente (`--base-url https://apihub.agnes-ai.com/v1`), com `--vision-model agnes-3.0-flash` para a camada de frames. O modo Gemini continua como padrão para entendimento nativo de vídeo.

## Limites observados

- Rate limit do tier gratuito da Agnes é apertado: rajadas de 4+ requisições seguidas retornam 429. Em produção, usar 1 req por vez com intervalo de ~15-20s, ou assinar Token Plan.
- Não foi testado: `agnes-image-*` em chamada real (doc indica geração apenas); modo `image-to-video` dos modelos de vídeo (geração, fora do escopo do Videolyzer).
