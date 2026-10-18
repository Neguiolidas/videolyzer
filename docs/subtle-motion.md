# Teste — Movimento Sutil (two-pass híbrido)

Data: 2026-10-18 · Script: `videolyzer.py`

## Problema

O modo nativo (Files API do Gemini) perdia movimento de pequena amplitude:
no teste comparativo anterior, o quique vertical sutil do quadrado azul foi
descrito como estático.

## Solução

Two-pass híbrido no modo arquivo local:
1. **Passada 1 (OpenCV, Farneback):** amostra pares de frames a 2fps, calcula
   magnitude média do fluxo por célula (grade 3x3), marca segmentos onde a
   célula mais ativa supera o ruído de fundo. Limiar robusto: mediana +
   4·1.4826·MAD (o desvio-padrão era inflado pelos próprios outliers).
   Também extrai o eixo dominante do movimento (|vx| vs |vy|).
2. **Passada 2 (Gemini, Files API):** analisa o vídeo inteiro normalmente,
   mas recebe os timestamps marcados + região + eixo como dica explícita
   no prompt ("NÃO diga que está estático nestes trechos").

Modo URL: sem detecção (exigiria download) — documentado no docstring.

## Vídeo de teste

Sintético, 20s, 640x480, 30fps, gerado por script (frames via OpenCV,
encode libx264):
- 00:00–00:08: quadrado vermelho 60px estático no centro
- 00:08–00:12: quique vertical sutil, ±12px, período 2s (seno)
- 00:12–00:20: estático de novo

Ground truth: movimento vertical em 00:08–00:12, nada antes ou depois.

## Resultados

| Etapa | Resultado |
|---|---|
| Detecção (passada 1) | `00:08–00:12, centro, eixo vertical, mag 1.428` — exato |
| Falso-positivo (vídeo 10s estático) | nenhum segmento marcado |
| Relatório final | menciona o movimento com timing exato (00:08–00:12), descreve como "translação oscilatória no eixo vertical (para cima e para baixo)" |
| Direção | correta após adicionar eixo dominante (antes, o modelo chutou "horizontal e vertical") |

## Limites honestos

- Movimento de amplitude muito pequena (< ~0.25 px/frame de magnitude média)
  continua abaixo do piso do detector.
- Em vídeo com movimento global constante (ex.: câmera em pan contínuo),
  o limiar relativo sobe e movimento sutil local pode não se destacar —
  o caso coberto é movimento sutil em footage calmo.
- Modo URL não tem a passada 1.
