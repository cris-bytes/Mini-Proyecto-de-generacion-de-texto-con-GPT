# Microcuentos en español con GPT

Mini-proyecto de generación de texto para el curso de PLN. Caso propio inspirado en la separación entre recuperación y generación del [material de Sesión 5](https://github.com/Ohtar10/icesi-nlp/tree/main/Sesion5).

**Entregable principal: [Microcuentos_GPT.ipynb](Microcuentos_GPT.ipynb)**. Incluye explicación del problema, EDA, implementación visible, entrenamiento desde cero, resultados, gráficos, generaciones y conclusiones.

**Validación realizada:** 24 celdas de código ejecutadas sin errores. Se repitió el entrenamiento desde cero y coincidieron exactamente las métricas y generaciones de los cuatro modelos principales. Ver [VALIDACION.md](VALIDACION.md).

El asistente produce borradores de microcuentos en cinco temas. Se comparan un bigrama, una GRU y dos GPT pequeños de distinta profundidad. Una demo recupera un inicio con BM25 y genera su continuación. No usa APIs ni modelos de OpenAI.

## Ejecutar

Python 3.9–3.12, CPU y conexión a Internet únicamente para instalar dependencias. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/ejecutar_notebook.py
```

El comando ejecuta todas las celdas en un kernel nuevo y guarda el notebook con salidas. El entrenamiento se repite desde cero. También puede abrirse con VS Code/Jupyter, seleccionar el intérprete `.venv` y usar **Restart Kernel and Run All**. Para usar JupyterLab, instalarlo aparte con `python -m pip install jupyterlab`.

Los datos son locales; no hay descargas de corpus, claves, servidor Ollama ni GPU obligatoria. El tiempo real por modelo aparece en el notebook y en `artifacts/resultados.csv`. Los checkpoints se generan en `artifacts/` y se omiten de Git por tamaño: el notebook los reconstruye.

## Qué contiene

| Archivo | Propósito |
|---|---|
| `Microcuentos_GPT.ipynb` | Entregable con código, resultados y narrativa |
| `Generacion_Texto.py` | Implementación reutilizable del corpus, modelos y demo |
| `data/familias.json` | 60 plantillas de microcuentos, creadas con asistencia de IA |
| `data/microcuentos.csv` | 960 variantes, con tema, familia y partición |
| `data/DATASHEET.md` | Procedencia, construcción, limitaciones y uso |
| `artifacts/` | Métricas, figuras, generaciones, conclusiones y manifiesto |
| `scripts/ejecutar_notebook.py` | Ejecución desde kernel limpio; falla ante errores |
| `scripts/construir_notebook.py` | Fuente editorial del notebook; regenerarlo elimina sus salidas |
| `requirements.txt` | Dependencias fijadas del entorno de referencia |

## Diseño experimental

- **Partición por familias:** 40 de entrenamiento, 10 de validación y 10 de prueba; 16 variantes por familia. Un split aleatorio por filas compartiría plantillas.
- **BPE de bytes:** vocabulario aprendido solo de train. Sin palabras de prueba convertidas a UNK.
- **Cuatro modelos:** bigrama suavizado, GRU-128, GPT de una capa y GPT de tres capas; atención causal verificada.
- **Entrenamiento:** 24 épocas, AdamW, reducción coseno del aprendizaje y checkpoint por NLL de validación.
- **Memorización:** se conserva además el GPT de tres capas de la época 24 para contrastarlo con su checkpoint temprano; este diagnóstico no participa en la selección principal.
- **Evaluación:** prueba reservada, perplejidad e intervalos bootstrap por familias; no se afirma significancia entre modelos.
- **Generación:** temperatura, top-k, top-p, ablación de temperatura, diversidad, terminación y coincidencias con train.
- **Recuperación:** BM25 sobre familias de entrenamiento; prefijo recuperado separado de la continuación generada.

Se fija la semilla y se usa CPU determinista. Se registran versiones y hashes. Otras plataformas o versiones pueden producir diferencias numéricas; la variación entre semillas de entrenamiento no se estudia en esta entrega.

## Alcance y autoría

El corpus es **sintético y redactado con asistencia de IA**. Las 960 filas no representan 960 estructuras independientes ni una muestra de literatura humana. Se declaran esta procedencia y la asistencia en la implementación. El grupo debe revisar y comprender el trabajo antes de entregarlo.

El modelo aprende patrones de un conjunto reducido y puede memorizar, mezclar situaciones o producir frases incompletas. Las métricas léxicas y la perplejidad no miden creatividad ni sustituyen evaluación humana. La demo es un prototipo educativo para producir borradores.

## Resultados de referencia

La GRU obtuvo perplejidad de prueba **89.61**, frente a **168.97** del bigrama. El GPT de tres capas seleccionado por validación obtuvo **130.64**. Su checkpoint final escribió frases más legibles, pero alcanzó perplejidad **953.00** en prueba y una coincidencia media de secuencias de cinco palabras de **94.1%** con entrenamiento en cinco generaciones. Esta coincidencia es una medida de reutilización de secuencias, no el porcentaje de relatos copiados íntegramente.

El caso muestra un resultado negativo relevante: más fluidez en un corpus pequeño y repetitivo puede ocultar memorización. La utilidad literaria del sistema no queda demostrada; el notebook explica por qué y propone cómo evaluar una versión con datos independientes.

## Rúbrica

| Criterio | Evidencia en el notebook |
|---|---|
| Notebook completo | §§1–12: problema, EDA, implementación, resultados y conclusiones |
| Reproducibilidad | §§2–4 y 12: datos, versiones, semillas, ejecución limpia y hashes |
| Generación de texto | §§5–8: entrenamiento de GPT, curvas, evaluación y muestras |
| Innovación | §§3–10: familias, BPE, profundidad, bootstrap, copia, BM25 y demo |

La organización permite usar este repositorio como entrega del curso y añadir actividades posteriores. La tabla describe evidencias, no garantiza una calificación.
