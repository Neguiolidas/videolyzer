# Teste Comparativo: frames vs nativo

Data: 2026-10-18 · Script: `videolyzer.py`

## Vídeo de teste

`test_motion2.mp4` (synthetic) (20s, 640x360, gerado sinteticamente via ffmpeg + Python).
Ground truth conhecido:

| Segmento | Tempo | Conteúdo |
|---|---|---|
| 1 | 00:00–00:07 | Quadrado VERMELHO desliza da ESQUERDA para a DIREITA |
| gap | 00:07–00:08 | Tela preta |
| 2 | 00:08–00:15 | Quadrado AZUL quica verticalmente no centro |
| 3 | 00:15–00:20 | Quadrado VERMELHO desliza da DIREITA para a ESQUERDA |

## Modo frames (antigo)

Extrai 2 frames (t=0s, t=15s) + áudio, envia como imagens.

**Resultado:**
- Cena 1 (00:00–00:07): vermelho esquerda→direita — CORRETO
- Cena 2 (00:07–00:15): "quadrado vermelho reverte e vai para a esquerda" — ERRADO.
  Nesse intervalo o que existe é o quadrado AZUL quicando. O modo frames
  alucinou continuidade do vermelho e não viu o azul em nenhum momento.
- Cena 3 (00:15–00:20): "vermelho vai para a direita" — DIREÇÃO ERRADA
  (o correto é direita→esquerda).

**Placar: 1/3 cenas corretas.** Perdeu um segmento inteiro e errou direção.

## Modo nativo (novo)

Upload do vídeo inteiro pela Files API do Gemini, modelo amostra sozinho.

**Resultado:**
- Cena 1 (00:00–00:07): vermelho movendo-se horizontalmente — CORRETO
- Cena 2 (00:08–00:14): quadrado azul no centro — CORRETO (detectou presença
  e timing; descreveu como estático, o quique sutil passou batido — parcial)
- Cena 3 (00:15–00:19): vermelho movendo-se horizontalmente — CORRETO

**Seção "Verificação temporal" (nova):**
- "O quadrado vermelho **desliza** da esquerda para a direita" ✓
- "O quadrado vermelho **desliza** da direita para a esquerda" ✓
- "Os eventos seguem uma lógica linear de alternância: movimento, pausa,
  movimento de retorno" ✓ (ordem temporal confirmada)

**Placar: 3/3 cenas corretas**, direções certas, ordem temporal confirmada.

## Conclusão

O modo nativo captura o que o modo frames perde:

1. **Segmentos intermediários**: frames a cada 15s pularam o quadrado azul
   por completo; o nativo detectou presença e timing corretos.
2. **Direção do movimento**: frames errou a direção do retorno (disse
   esquerda→direita, era direita→esquerda); o nativo acertou.
3. **Alucinação**: com só 2 frames, o modelo preencheu as lacunas inventando
   continuidade ("o vermelho reverte e volta"); o nativo não precisou inventar.
4. **Verbos de movimento**: a seção de verificação temporal força descrição
   de ação em vez de estado estático, e o nativo a preenche corretamente.

Limite honesto: o quique vertical sutil do quadrado azul foi descrito como
estático. Movimento de pequena amplitude ainda pode passar batido.
