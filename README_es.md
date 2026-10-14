🇬🇧 [English](README.md) · 🇪🇸 [Español](README_es.md)

# lyric-emotion — análisis de emociones en letras de canciones

**¿The Cure y The Smiths sonaron realmente tan distintos, emocionalmente, como
sugieren sus reputaciones?** Ambos son pilares del post-punk/rock gótico
británico de los años 80, y ambos terminan reducidos a "deprimentes" en la
descripción casual. Este proyecto responde esa pregunta con un modelo de
emociones fino y reproducible, en vez de apoyarse en impresiones.

La mayoría de los proyectos de portfolio de "análisis de sentimiento en letras"
usan un clasificador de tweets de 3 clases ya entrenado. Este, en cambio:

1. Ajusta (fine-tuning) su propio modelo — 28 emociones (taxonomía GoEmotions)
   más un espacio continuo de valencia/activación/dominancia (VAD) — sobre
   corpus redistribuibles (GoEmotions, EmoBank), así que cualquiera puede
   reproducir el entrenamiento base sin API key.
2. Adapta ese modelo al dominio de las letras con datasets de etiquetado débil
   (MoodyLyrics, Deezer DMDD).
3. Está pensado para validarse contra un gold set anotado a mano — 80
   canciones, guías publicadas y un protocolo de validación fijado antes de
   anotar, con consistencia intra-anotador reportada. **Esa anotación sigue
   pendiente** (ver Limitaciones): todos los resultados de abajo son previos
   al gold set.
4. Lo aplica a una pregunta real: cómo cambió la paleta emocional de The Cure
   y The Smiths a lo largo de sus carreras y álbumes, con incertidumbre y
   tamaños de muestra reportados en cada paso.

## Herramientas 🛠️

- Python, pandas, NumPy
- PyTorch, Hugging Face Transformers (fine-tuning de ModernBERT-base)
- GoEmotions, EmoBank, MoodyLyrics, Deezer DMDD, léxico NRC-VAD
- API de Genius (letras descargadas en local, nunca redistribuidas)
- Streamlit (dashboard y herramienta de anotación manual)
- matplotlib
- uv, ruff, mypy, pytest, GitHub Actions

## Skills 🧠

- Fine-tuning de transformers: clasificación multietiqueta y regresión multisalida
- Adaptación de dominio con etiquetas débiles
- Evaluación contra baselines públicos, reportando con honestidad cuando gana el baseline
- Diseño de anotación: guías y protocolo de validación fijados antes de anotar
- Inferencia estadística sobre datos agrupados (permutación y remuestreo por álbum, corrección FDR)
- Empaquetar un proyecto de ML como paquete instalable con CLI, tests y CI
- Comunicar resultados con un dashboard interactivo

## Datos 📎

| etapa | fuente | tamaño | ¿redistribuible? |
|---|---|---|---|
| Cabeza de emociones (etapa 1) | GoEmotions | 43.410 train / 5.426 val | sí (Apache-2.0) |
| Cabeza VAD (etapa 1) | EmoBank | 8.062 train / 999 dev | sí (CC BY-SA 4.0) |
| Adaptación de dominio (etapa 2) | MoodyLyrics + muestra de DMDD, letras vía Genius | 4.913 canciones con etiquetado débil | no — solo IDs/etiquetas, las letras se descargan por separado |
| Corpus del caso de estudio | Genius, deduplicado a tomas de estudio | The Cure 256 canciones / The Smiths 75 canciones | no — los términos de Genius no permiten redistribuir |
| Gold set | anotado a mano por mí | 80 canciones muestreadas, anotación pendiente | n/a |

Términos de licencia completos y uso de cada fuente: `THIRD_PARTY_NOTICES.md`.

## Decisiones de diseño 🧭

Dos fine-tunes independientes de ModernBERT-base en lugar de un modelo
multitarea — más simple de depurar; multitarea queda como ablación opcional:

- **Cabeza de emociones**: clasificación multietiqueta de 28 clases, BCE
  ponderada (etiquetas raras como `grief`/`relief` tienen <200 positivos entre
  43k filas), con umbrales por clase ajustados en validación en vez de un 0,5
  global.
- **Cabeza VAD**: regresión de 3 salidas (valencia/activación/dominancia), pérdida MSE.
- **Etapa 2 (adaptación de dominio)**: continúa el checkpoint VAD con letras de
  etiquetado débil, solo valencia/activación — las etiquetas débiles (derivadas
  de tags de Last.fm) no tienen señal de dominancia, así que esa salida queda
  con los pesos de la etapa 1.
- **Chequeo de consistencia sin etiquetas**: las 28 probabilidades de emoción
  de cada canción se proyectan a VAD con centroides del léxico NRC-VAD y se
  correlacionan con la salida de la cabeza de regresión sobre las mismas
  canciones — una verificación que no necesita etiquetas del gold set.
- **Puntaje del caso de estudio**: la cabeza de emociones puntúa cada canción
  por secciones; la cabeza VAD puntúa la letra completa (como fue entrenada).
  Antes se filtran covers, tomas en vivo y versiones de otros artistas.
- **La inferencia cuenta álbumes, no canciones**: las canciones de un álbum
  suenan parecido, así que las diferencias se testean permutando/remuestreando
  álbumes enteros y la separabilidad se valida cruzadamente sobre álbumes que
  el modelo no vio.

## Resultados 📊

Previos al gold set: todos los números de abajo son provisorios.

- **Cabeza de emociones vs. baseline** (split de test de GoEmotions, umbrales
  ajustados en validación): `cirimus/modernbert-base-go-emotions` le gana
  levemente a este modelo (0,5531 vs. 0,5127 de F1 macro — mismo modelo base y
  dataset). Se reporta con honestidad: el valor acá está en la adaptación de
  dominio y la cabeza VAD, no en superar a un modelo equivalente en su propio
  benchmark.
- **Cabeza VAD vs. baseline** (split de test de EmoBank): supera claramente a un
  baseline de léxico NRC-VAD en las tres dimensiones (CCC 0,71/0,47/0,32 vs.
  0,28/0,17/0,14 para valencia/activación/dominancia).
- **Etapa 1 vs. etapa 2, en igualdad de condiciones**: evaluado sobre el mismo
  split de test de DMDD con agregación por secciones, el checkpoint adaptado
  (`vad_domain`) supera al de la etapa 1 en las dos dimensiones para las que se
  adaptó (CCC de valencia 0,234 vs. 0,068; de activación 0,040 vs. 0,017), así
  que el caso de estudio corre sobre `vad_domain` — decidido por
  `docs/vad_checkpoint_comparison.md`, no a mano. La dominancia sigue saliendo
  de los pesos no adaptados de la etapa 1 (las etiquetas débiles no tienen
  señal de dominancia).
- **Chequeo de consistencia**: las probabilidades de emoción proyectadas a VAD
  correlacionan con la cabeza de regresión con r de Pearson 0,67 (valencia),
  0,42 (activación), 0,24 (dominancia) — las dos cabezas coinciden en general,
  más en valencia y menos en dominancia.
- **The Cure vs. The Smiths, 1982-1987** (los años en que ambos estuvieron
  activos; `findings.json`; 67 y 66 canciones, 8 álbumes de estudio más grupos
  anuales de singles y lados B, 18 grupos en total): no hay una diferencia clara
  en oscuridad (delta de Cliff -0,06, intervalo del 95 % remuestreado por álbum
  de -0,30 a 0,13), pero el perfil de 28 emociones distingue a las bandas en
  álbumes no vistos (AUC validado cruzadamente 0,84, p de permutación a nivel
  álbum = 0,005; 0,83 sin las emociones relacionadas con preguntas). Una
  familia de emociones se llama "clara" solo cuando el p ajustado por FDR a
  nivel álbum es menor a 0,05 y el intervalo del 95 % queda más allá del menor
  efecto de interés (0,147): "Alegría y diversión" (mayor en The Cure; delta
  0,46, intervalo 0,25 a 0,67) lo cumple. "Cuestionamiento" (mayor en The
  Smiths, sigue a las líneas que terminan en "?"; delta -0,35, p ajustado =
  0,035) es sugestivo: su intervalo, de -0,57 a -0,14, apenas alcanza el
  umbral. La valencia es el promedio de tres semillas de entrenamiento de
  `vad_domain`, porque una sola semilla mueve la brecha entre -0,02 y -0,23
  durante 1980-89 (`docs/vad_domain_ablation.md`). Activación y dominancia no
  se sostuvieron (activación r = 0,135 contra DMDD) y no se reportan.

## Dashboard 📈

`app/` es una app Streamlit autocontenida que renderiza un bundle chico
(`app/data/`, generado por `lyric-emotion export-app`; solo puntajes, sin
letras). Su pieza central es un mapa de dos ejes: un punto por canción, brillo
en un eje y "sensación general" (la mejor combinación fuera de muestra de los
puntajes de emoción para distinguir a las bandas) en el otro. Las figuras
estáticas están en `output/figures/`.

## Quick start 🚀

```bash
uv sync
cp .env.example .env   # completar GENIUS_ACCESS_TOKEN y HF_TOKEN en local, nunca commitear
uv run pytest
```

Los datos de entrenamiento y las cabezas de emociones/VAD (fases 1-2) no
requieren credenciales. Descargar letras para el caso de estudio o el set de
adaptación de dominio (fases 3-4) requiere un token gratuito de la API de Genius.

```bash
uv run lyric-emotion build-training              # Fase 1: GoEmotions + EmoBank
uv run lyric-emotion train --head emotions       # Fase 2
uv run lyric-emotion train --head vad            # Fase 2
uv run lyric-emotion evaluate-baselines          # Fase 2/4 baselines + chequeo de consistencia
uv run lyric-emotion fetch                       # Fase 3: corpus The Cure / The Smiths
uv run lyric-emotion build-domain-training       # Fase 4: MoodyLyrics + muestra de DMDD
uv run lyric-emotion train --head vad-domain     # Fase 4 etapa 2
uv run lyric-emotion analyze && uv run lyric-emotion report   # Fase 5
uv run lyric-emotion gold-sample                              # Fase 4: sortea las 80 canciones
uv run streamlit run annotation/annotate.py                   # anotarlas (solo local)
uv run lyric-emotion evaluate-gold                            # chequeo humano -> dashboard
uv run lyric-emotion export-app                               # Fase 6: bundle
uv run streamlit run app/streamlit_app.py                     # dashboard
```

## Limitaciones ⚠️

- **La anotación del gold set sigue pendiente** — toda comparación de arriba es
  contra un baseline público o contra las propias etiquetas débiles de
  entrenamiento del modelo, no contra verdad verificada a mano. Los números son
  provisorios.
- Las etiquetas débiles de dominio (MoodyLyrics/DMDD) se derivan de tags de
  Last.fm, no de anotación directa — limitan lo que puede demostrar la etapa 2.
- Solo señal de letra: la valencia y el ánimo que transmiten la melodía, el
  tempo y la instrumentación no se modelan.
- Solo inglés; metáfora, ironía y narradores ficticios no se tratan
  específicamente — una lectura literal de letras oscuras puede sobreestimar el
  afecto negativo.
- La deduplicación a tomas de estudio no resuelve todos los casos borde de
  en vivo/reediciones; ver `docs/dedup_report.md`.
- Sin afirmaciones causales: las diferencias temporales y entre artistas se
  reportan como asociaciones con intervalos de confianza y tamaños de muestra,
  no como causas.

Lo que no se publica acá: el texto completo de las letras, los tokens de API y
las respuestas crudas de la API nunca se commitean — ver `.gitignore` y
`THIRD_PARTY_NOTICES.md`. Solo son públicos el código, las configs, los IDs, los
puntajes agregados, las figuras y las guías de anotación.

## Licencia

Código: MIT (ver `LICENSE`). Los datasets y modelos de terceros mantienen sus
propias licencias — ver `THIRD_PARTY_NOTICES.md`.
