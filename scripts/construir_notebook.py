"""Genera el notebook con el código del modelo visible, no oculto en imports."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))
source = (ROOT / "Generacion_Texto.py").read_text()
sections = {}
for part in source.split("# %% ")[1:]:
    name, body = part.split("\n", 1)
    sections[name.strip()] = body.strip()

md(r"""
# Laboratorio de microcuentos en español: GPT, memoria y creatividad

**Mini-proyecto de generación de texto · Procesamiento del Lenguaje Natural**
Cuenta del repositorio: **cris-bytes** · Caso propio: cinco universos de ficción breve.

### 1. Problema y alcance
Quien empieza a escribir suele necesitar una primera idea, no una historia terminada. Proponemos un
**asistente experimental de borradores**: recibe un tema y produce un microcuento; opcionalmente
recupera una primera frase del corpus para continuarla. La persona decide qué conservar y reescribir.
Una propuesta útil debería mantener un tema, formar frases comprensibles y evitar reproducir relatos enteros.

Un GPT causal es adecuado porque aprende $p(x_t\mid x_{<t},tema)$ y puede producir continuaciones nuevas.
Compararlo con un bigrama y una GRU permite preguntar si más contexto y atención justifican su coste.
La recuperación facilita un punto de partida concreto, aunque también puede favorecer la copia.

**Alcance honesto:** entrenamos desde cero un Transformer pequeño de tipo GPT, no un modelo de OpenAI
ni un asistente con conocimientos generales. El corpus es **sintético, redactado con asistencia de IA**:
60 plantillas narrativas y 16 variantes por plantilla. No representa la literatura en español ni equivale
a 960 relatos independientes. Las salidas son borradores experimentales, no cuentos de calidad garantizada.

### Relación con la sesión y aportes propios
Los [notebooks de Sesión 5](https://github.com/Ohtar10/icesi-nlp/tree/main/Sesion5) usan WikiHow,
Ollama y recuperación; expresamente no entrenan modelos. Aquí cambiamos tanto los datos como el problema
y añadimos entrenamiento para cubrir la rúbrica. Conservamos la separación **recuperador → contexto → generación**,
adaptada a un LM de continuación, y mostramos las fuentes del prefijo.

| Aspecto | Desarrollo de este caso |
|---|---|
| EDA | Longitudes, balance, léxico, cobertura BPE, duplicación y fuga por familias |
| Modelado | Bigrama suavizado, GRU, GPT de una y tres capas |
| Experimentos | Curvas, selección por validación, prueba reservada, temperatura, top-k y top-p |
| Innovación | Bootstrap por familias, auditoría de copia, BM25 y demo reproducible |
| Reproducibilidad | Datos locales, semillas, CPU, versiones y entrenamiento desde cero |

### Preguntas que contrastaremos
1. ¿Los modelos neuronales reducen la perplejidad respecto del bigrama en familias no vistas?
2. ¿Tres capas mejoran la generalización frente a una capa, con igual número de épocas y datos?
3. ¿Cambiar el muestreo aumenta diversidad a costa de coherencia o terminación?
4. ¿Recuperar un inicio aumenta la afinidad temática y el riesgo de copiar?

**Criterio técnico previo:** seleccionar checkpoints y arquitectura por menor NLL de validación.
La calidad narrativa se examina por separado; una menor perplejidad no demuestra creatividad.
""")
md("""
### 2. Ejecución desde cero
Desde la raíz del repositorio: `python -m pip install -r requirements.txt`, abrir este notebook con ese
intérprete y ejecutar **Restart Kernel and Run All**. Alternativa: `python scripts/ejecutar_notebook.py`.
Se requiere Python 3.9–3.12 y aproximadamente unos minutos de CPU; el tiempo real se registra abajo.
No se necesitan claves, Ollama, GPU ni descargar modelos. Después de instalar dependencias, funciona sin red.
Las celdas entrenan de nuevo; no usan resultados guardados como sustituto del entrenamiento.
Los checkpoints se guardan localmente en `artifacts/`, pero no son necesarios para reproducir el trabajo.
""")
code("""
from pathlib import Path
import os
ROOT = Path.cwd()
assert (ROOT / "data/familias.json").exists(), "Abra el notebook desde la raíz del repositorio."
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
(ROOT / "artifacts").mkdir(exist_ok=True)
""")
code(sections["imports"])
code("""
seed_all()
import importlib.metadata as metadata
runtime = {"python": platform.python_version(), "plataforma": platform.platform(),
           "semilla": SEED, "dispositivo": "cpu", "threads": torch.get_num_threads(),
           "versiones": {p: metadata.version(p) for p in ["torch", "numpy", "pandas", "tokenizers", "matplotlib"]}}
print(json.dumps(runtime, ensure_ascii=False, indent=2))
(ROOT / "artifacts/runtime.json").write_text(json.dumps(runtime, indent=2))
""")
md("""
### 3. Datos, procedencia y prevención de fuga
`data/familias.json` contiene los textos base originales de este ejercicio, redactados con ayuda de IA.
Cada relato tiene un inicio, una situación extraña y una resolución o imagen final. Se expanden únicamente
nombre, lugar y momento; **no se afirma que las variantes sean muestras independientes**.
Se conservaron tildes, puntuación y mayúsculas porque son parte de la escritura que queremos modelar.

La partición se decide por **familia**, estratificada por tema: 8 familias para entrenamiento, 2 para
validación y 2 para prueba en cada tema. Ninguna variante de una familia puede aparecer en otro conjunto.
Los motivos compartidos —cartas, ventanas, recuerdos— sí atraviesan conjuntos: la separación reduce la fuga
de plantillas, pero no garantiza independencia semántica. Esta limitación se conserva al interpretar resultados.
""")
code(sections["corpus"])
code("""
df = build_corpus(ROOT)
display(df.groupby(["split", "tema"]).agg(relatos=("id", "size"), familias=("familia", "nunique")))
display(df.groupby("split").n_palabras.describe().round(1))
display(df.drop_duplicates("tema")[["tema", "texto"]].style.set_properties(**{"white-space": "normal"}))
print("Nulos:", int(df.isna().sum().sum()), "· duplicados exactos:", int(df.texto.duplicated().sum()))
""")
code("""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for split, group in df.groupby("split"):
    ax[0].hist(group.n_palabras, bins=12, alpha=.45, label=split)
ax[0].set(xlabel="Palabras por relato", ylabel="Relatos", title="Longitud por partición")
ax[0].legend()
df.groupby("tema").size().plot.bar(ax=ax[1], color="#327a88", rot=25)
ax[1].set(title="Balance construido por diseño", ylabel="Relatos", xlabel="Tema")
counts = Counter(w for t in df[df.split == "train"].texto for w in words(t))
freqs = sorted(counts.values(), reverse=True)
ax[2].loglog(range(1, len(freqs)+1), freqs, color="#a3485b")
ax[2].set(xlabel="Rango", ylabel="Frecuencia", title="Frecuencia de palabras en train")
fig.tight_layout()
fig.savefig(ROOT / "artifacts/eda.png", bbox_inches="tight")
plt.show()
display(pd.DataFrame(counts.most_common(15), columns=["palabra", "frecuencia"]))
""")
code("""
# Un split aleatorio por filas dejaría familias presentes en ambos lados.
perm = np.random.default_rng(SEED).permutation(len(df))
cut = int(.8*len(df))
random_train, random_eval = df.iloc[perm[:cut]], df.iloc[perm[cut:]]
leak_rate = random_eval.familia.isin(random_train.familia).mean()
real_overlap = set(df[df.split == "train"].familia) & set(df[df.split != "train"].familia)
assert not real_overlap
display(Markdown(f"**Hallazgo de EDA:** un split por filas compartiría familias en el **{leak_rate:.1%}** "
                "de las filas de evaluación. La partición por familias evita ese atajo. "
                f"Hay **{len(df)} variantes**, pero solo **{df.familia.nunique()} estructuras base**. "
                "El balance es artificial; no describe la frecuencia natural de los temas."))
""")
md("""
### 4. Tokenización y diseño de la tarea
Usamos BPE de bytes con 768 tokens: aprende fragmentos frecuentes **solo de entrenamiento** y puede
representar texto UTF-8 desconocido sin convertir palabras de prueba en un token UNK.
La secuencia es `<bos> <tema> texto <eos>`. La etiqueta del tema es una condición conocida:
su predicción y el padding se excluyen de la pérdida. Sí se evalúa la predicción del fin del relato.
El vocabulario y los targets son iguales en todos los modelos, por lo que sus perplejidades son comparables.
""")
code("""
tok = fit_tokenizer(df, ROOT)
encoded = {s: encode_rows(df[df.split == s], tok) for s in ["train", "val", "test"]}
df["tokens_bpe"] = [len(tok.encode(t).ids) for t in df.texto]
assert max(len(x) for rows in encoded.values() for x in rows) <= 256
display(df.groupby("split")[["n_palabras", "tokens_bpe"]].mean().round(2))
train_vocab = set(w for t in df[df.split == "train"].texto for w in words(t))
lexical_oov = {s: np.mean([w not in train_vocab for t in df[df.split == s].texto for w in words(t)])
               for s in ["val", "test"]}
print("Vocabulario BPE:", tok.get_vocab_size(), "· máxima longitud:", df.tokens_bpe.max())
print("Fracción de palabras no vistas (BPE sí las representa):", lexical_oov)
print("Ejemplo de segmentación:", tok.encode("Una brújula recordaba el mar.").tokens)
display(Markdown("**Decisión:** los relatos completos caben en 256 tokens; no se truncaron finales. "
                "Las palabras nuevas se descomponen en fragmentos, aunque poder representarlas no implica "
                "haber aprendido a utilizarlas. Los temas y variantes balanceados simplifican mucho la tarea."))
""")
md(r"""
### 5. Arquitecturas y controles
El bigrama estima la transición entre dos tokens con suavizado aditivo $\alpha=0.1$; no tiene memoria larga.
La GRU mantiene un estado recurrente. El GPT usa embeddings de token y posición, atención causal multicefálica,
residuales, normalización y MLP. No utiliza encoder ni atención bidireccional.

La pérdida es $\mathcal L=-\sum_t\log p(x_t\mid x_{<t},tema)/N$ y la perplejidad es $\exp(\mathcal L)$.
Las capas GPT solo ven posiciones pasadas; el siguiente token aparece únicamente en el objetivo.
La tabla de parámetros muestra que la comparación **no iguala capacidad ni FLOPs**. La ablación de profundidad
mantiene ancho, datos, optimizador y épocas, pero cambia el número de parámetros.
""")
code(sections["architectures"])
code(sections["training"])
code(sections["generation"])
code("""
# Verificación de causalidad: cambiar el sufijo no debe alterar predicciones del prefijo.
seed_all()
untrained = TinyGPT(tok.get_vocab_size(), layers=1).eval()
x = torch.tensor([encoded["train"][0][:-1]])
changed = x.clone()
changed[:, 12:] = (changed[:, 12:]+1) % tok.get_vocab_size()
with torch.no_grad():
    assert torch.allclose(untrained(x)[:, :12], untrained(changed)[:, :12], atol=1e-6)
print("Máscara causal verificada.")
print("GPT antes de entrenar:", generate(untrained, tok, "mar", max_new_tokens=45)["texto"])
display(pd.DataFrame(CONFIGS))
""")
md("""
### 6. Entrenamiento real y selección por validación
Entrenamos 24 épocas, lotes de 32 relatos, AdamW, decaimiento de pesos 0.01, recorte de gradiente a 1
y aprendizaje inicial 0.002 con descenso coseno. En GPT usamos dropout 0.1. Todos reciben los mismos relatos
y el mismo orden de lotes. Guardamos el checkpoint con menor NLL de validación de cada modelo.
Una sola semilla de entrenamiento permite una comparación reproducible, pero no estima la variabilidad
entre inicializaciones; las semillas de generación posteriores no sustituyen ese experimento.
""")
code("""
models_trained, history, validation = train_experiments(encoded, tok.get_vocab_size(), ROOT, epochs=24)
primary_validation = validation[validation.modelo != "GPT-3c-final"]
best_name = primary_validation.sort_values("val_nll").iloc[0].modelo
best_gpt_name = primary_validation[primary_validation.modelo.str.startswith("GPT")].sort_values("val_nll").iloc[0].modelo
best_gpt = models_trained[best_gpt_name]
display(validation.sort_values("val_nll").round(4))
print("Ganador por validación:", best_name, "· GPT para la demo:", best_gpt_name)
""")
code("""
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, h) in zip(axs, history.groupby("modelo", sort=False)):
    ax.plot(h.epoca, h.train_nll, label="Train (durante la época)")
    ax.plot(h.epoca, h.val_nll, label="Validación (eval)")
    selected = int(validation[validation.modelo == name].mejor_epoca.iloc[0])
    ax.axvline(selected, ls="--", c="gray", label=f"Checkpoint {selected}")
    ax.set(title=name, xlabel="Época", ylabel="NLL / token")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "artifacts/curvas_entrenamiento.png", bbox_inches="tight")
plt.show()
gap = validation.assign(brecha=validation.val_nll-validation.train_nll)
display(gap[["modelo", "mejor_epoca", "train_nll", "val_nll", "brecha"]].round(3))
display(Markdown(f"**Lectura:** se eligió **{best_name}** por validación. Una pérdida de entrenamiento menor "
                "que la de validación es compatible con memorización de estructuras. "
                "La curva de train se mide durante las actualizaciones y con dropout; la tabla anterior "
                "reevalúa el checkpoint en modo eval, por eso las cifras no tienen que coincidir."))
""")
md("""
### 7. Prueba reservada e incertidumbre
Solo después de fijar los checkpoints calculamos métricas de prueba. Reportamos NLL ponderada por tokens,
perplejidad e intervalos percentiles del 95% con 1.000 remuestreos de **familias completas**.
Diez familias de prueba son pocas: los intervalos reflejan sensibilidad al corpus, no incertidumbre por semilla
ni evidencia de superioridad universal. No cambiamos hiperparámetros a partir de esta tabla.
""")
code("""
test_meta = df[df.split == "test"].reset_index(drop=True)
test_tables, scores = {}, []
for name, model in models_trained.items():
    table = evaluate(model, encoded["test"])
    table["familia"], table["tema"] = test_meta.familia, test_meta.tema
    test_tables[name] = table
    nll = global_nll(table)
    low, high = bootstrap_family(table)
    scores.append(dict(modelo=name, test_nll=nll, perplexity=math.exp(nll), ci_low=low, ci_high=high))
results = validation.merge(pd.DataFrame(scores), on="modelo")
results.to_csv(ROOT / "artifacts/resultados.csv", index=False)
display(results.sort_values("val_nll").round(3))
fig, ax = plt.subplots(figsize=(8, 4))
r = results.set_index("modelo")
ax.errorbar(r.index, r.perplexity, yerr=[r.perplexity-r.ci_low, r.ci_high-r.perplexity], fmt="o", capsize=5)
ax.set(ylabel="Perplejidad de prueba", title="IC 95% por remuestreo de familias")
plt.show()
by_theme = pd.DataFrame({name: table.groupby("tema").apply(
    lambda g: math.exp(global_nll(g)), include_groups=False) for name, table in test_tables.items()})
display(by_theme.round(2))
""")
md("""
### 8. Muestreo: temperatura, top-k y nucleus
Usamos el GPT seleccionado por validación. Temperatura divide los logits antes del softmax: valores bajos
concentran probabilidades. Top-k limita candidatos; top-p conserva una masa acumulada de probabilidad.
Comparamos configuraciones predefinidas para los cinco temas y dos semillas de generación. No escogemos
ejemplos por su apariencia. `distinct2` mide diversidad de bigramas dentro de cada salida; `copia5` es la
fracción de sus secuencias de cinco palabras que también aparecen en train. Esta última detecta coincidencias,
no plagio ni copia literal de un relato completo. La afinidad temática es un indicador léxico imperfecto.
""")
code("""
train_texts = df[df.split == "train"].texto.tolist()
decoders_config = [
    {"config": "fría", "temperature": .55, "top_k": 20, "top_p": .9},
    {"config": "equilibrada", "temperature": .85, "top_k": 40, "top_p": .95},
    {"config": "caliente", "temperature": 1.15, "top_k": 0, "top_p": 1.},
    {"config": "nucleus", "temperature": .85, "top_k": 0, "top_p": .8},
]
generated = []
for cfg in decoders_config:
    for theme in THEMES:
        for seed in [17, 29]:
            output = generate(best_gpt, tok, theme, seed=seed,
                              **{k:v for k,v in cfg.items() if k != "config"})
            generated.append(dict(config=cfg["config"], tema=theme, seed=seed, **output,
                                  **generation_metrics(output["texto"], train_texts, theme)))
samples = pd.DataFrame(generated)
samples.to_csv(ROOT / "artifacts/generaciones.csv", index=False)
sampling_summary = samples.groupby("config")[["palabras", "distinct2", "copia5", "jaccard_max", "tema_lexico", "eos"]].mean()
display(sampling_summary.round(3))
sampling_summary[["distinct2", "copia5", "tema_lexico", "eos"]].plot.bar(figsize=(10, 4), rot=0)
plt.ylabel("Proporción media")
plt.title("Diversidad, coincidencia con train, afinidad temática y terminación")
plt.tight_layout()
plt.savefig(ROOT / "artifacts/muestreo.png", bbox_inches="tight")
plt.show()
for row in samples[(samples.tema == "mar") & (samples.seed == 17)].itertuples():
    display(Markdown(f"**{row.config} · EOS={row.eos}**\\n\\n{row.texto}"))
""")
md("""
**Cómo interpretar:** un `distinct2` alto puede corresponder a palabras incoherentes; no es una puntuación
de calidad. La terminación EOS informa si el modelo cerró el relato antes del límite de tokens, no si el
final es satisfactorio. La comparación cambia algunos parámetros a la vez, por lo que describe recetas
de muestreo y no identifica el efecto causal aislado de cada parámetro. La siguiente ablación sí mantiene
top-k y top-p constantes al variar solo temperatura.
""")
code("""
temperature_rows = []
for temp in [.55, .85, 1.15]:
    for theme in THEMES:
        out = generate(best_gpt, tok, theme, temperature=temp, top_k=40, top_p=.95, seed=43)
        temperature_rows.append(dict(temperatura=temp, tema=theme, **out,
                                     **generation_metrics(out["texto"], train_texts, theme)))
temperature_samples = pd.DataFrame(temperature_rows)
temperature_samples.to_csv(ROOT / "artifacts/ablacion_temperatura.csv", index=False)
display(temperature_samples.groupby("temperatura")[["distinct2", "copia5", "tema_lexico", "eos"]].mean().round(3))
""")
md("""
### Diagnóstico adicional: checkpoint temprano frente a entrenamiento prolongado
La selección por validación puede detenerse antes de que la generación suene fluida, especialmente con
solo 40 estructuras de entrenamiento. Conservamos también el GPT de tres capas de la época 24 como
**diagnóstico de memorización**, excluido de la selección principal. No se propone como ganador de generalización.
Comparamos el mejor checkpoint de esa misma arquitectura contra el final, con temas y semilla iguales.
Una salida más legible que copia train no prueba aprendizaje creativo ni mejora en familias nuevas.
""")
code("""
late_rows = []
for name in ["GPT-3c", "GPT-3c-final"]:
    for theme in THEMES:
        out = generate(models_trained[name], tok, theme, temperature=.85, seed=17)
        late_rows.append(dict(modelo=name, tema=theme, **out,
                              **generation_metrics(out["texto"], train_texts, theme)))
late_samples = pd.DataFrame(late_rows)
late_samples.to_csv(ROOT / "artifacts/diagnostico_memorizacion.csv", index=False)
display(late_samples.groupby("modelo")[["distinct2", "copia5", "tema_lexico", "eos"]].mean().round(3))
for row in late_samples[late_samples.tema == "mar"].itertuples():
    display(Markdown(f"**{row.modelo}**\\n\\n{row.texto}"))
""")
md("""
### 9. Recuperación y generación: adaptación del ejemplo de clase
Indexamos **solo las 40 familias de entrenamiento**, una variante por familia, con BM25. Su recuperación es
léxica: no entiende sinónimos como un encoder semántico. Por eso puede fallar incluso cuando existe un relato útil.
Evaluamos hit@1 y hit@3 **temáticos**, no relevancia humana ni recuperación de un documento exacto.

El GPT pequeño no fue entrenado para obedecer instrucciones de chat. Recuperamos la primera oración de
un relato y la usamos como prefijo para continuarlo. Es generación condicionada por texto recuperado,
inspirada en la organización de RAG de la sesión; **no reproduce el RAG original entrenado conjuntamente**.
Separamos prefijo reutilizado y continuación nueva, y mostramos la fuente del prefijo.
""")
code(sections["retrieval"])
code("""
retriever = BM25(df[df.split == "train"])
queries = [
    ("memoria", "una carta con recuerdos de la infancia"),
    ("memoria", "una fotografía y el recuerdo de una casa"),
    ("ciudad", "un bus recorre las calles del barrio"),
    ("ciudad", "ventanas y edificios en una ciudad"),
    ("naturaleza", "un árbol crece junto al río"),
    ("naturaleza", "las flores y las semillas de un jardín"),
    ("tecnologia", "un robot aprende a hablar con una máquina"),
    ("tecnologia", "una pantalla y un servidor guardan mensajes"),
    ("mar", "un barco y una isla en el océano"),
    ("mar", "una carta llega con las olas del mar"),
]
retrieval_scores = []
for theme, query in queries:
    hits = retriever.search(query)
    retrieval_scores.append(dict(tema=theme, consulta=query,
        hit1=float(len(hits)>0 and hits.iloc[0].tema == theme), hit3=float(theme in hits.tema.values)))
retrieval_results = pd.DataFrame(retrieval_scores)
retrieval_results.to_csv(ROOT / "artifacts/recuperacion.csv", index=False)
display(retrieval_results)
print("Promedio:", retrieval_results[["hit1", "hit3"]].mean().to_dict())
display(retriever.search("una carta llega con las olas del mar")[["familia", "tema", "score", "texto"]])
""")
code("""
# Evaluar SOLO la continuación: el prefijo recuperado es copia explícita por diseño.
retrieval_generations = []
for theme, query in queries:
    for retrieve in [False, True]:
        out = demo(best_gpt, tok, retriever, theme, query, retrieve, seed=17)
        retrieval_generations.append(dict(tema=theme, consulta=query, recuperacion=retrieve, **out,
            **generation_metrics(out["continuacion"], train_texts, theme)))
rag_samples = pd.DataFrame(retrieval_generations)
rag_samples.to_json(ROOT / "artifacts/demo_recuperacion.json", orient="records", force_ascii=False, indent=2)
display(rag_samples.groupby("recuperacion")[["distinct2", "copia5", "tema_lexico", "eos"]].mean().round(3))
example = rag_samples[(rag_samples.tema == "mar") & rag_samples.recuperacion].iloc[0]
print("Prefijo recuperado:", example.prefijo_recuperado)
print("Continuación generada:", example.continuacion)
print("Fuente del prefijo:", example.fuentes)
""")
md("""
**Límite del experimento:** sin recuperación, el modelo recibe solo el tema; con recuperación también recibe
el prefijo. Esta diferencia de información forma parte de la intervención. No demuestra que BM25 supere
a cualquier prefijo humano. Las dos consultas de un tema comparten la misma salida sin recuperación
con esta semilla; no son réplicas independientes. No calculamos significancia sobre esas filas duplicadas.
La fuente identifica material de inspiración y no verifica las afirmaciones ficticias generadas.

### 10. Demo editable
Cambie tema, idea, temperatura y semilla. La idea se utiliza para buscar el prefijo cuando `recuperar=True`;
sin recuperación, la condición del generador es únicamente el tema. Una consulta sin coincidencias devuelve
una lista de fuentes vacía y genera desde el tema.
""")
code("""
MI_TEMA = "naturaleza"
MI_IDEA = "un árbol y una semilla junto al río"
MI_TEMPERATURA = .85
MI_SEMILLA = 29
mi_cuento = demo(best_gpt, tok, retriever, MI_TEMA, MI_IDEA,
                recuperar=True, temperature=MI_TEMPERATURA, seed=MI_SEMILLA)
display(Markdown("**Prefijo:** " + mi_cuento["prefijo_recuperado"]))
display(Markdown("**Continuación:** " + mi_cuento["continuacion"]))
print("Fuentes:", mi_cuento["fuentes"])
# Modo de contraste: mismo prefijo, checkpoint final susceptible de memorizar.
mi_cuento_final = demo(models_trained["GPT-3c-final"], tok, retriever, MI_TEMA, MI_IDEA,
                      recuperar=True, temperature=MI_TEMPERATURA, seed=MI_SEMILLA)
display(Markdown("**Contraste con época 24 (riesgo de memorización):** " + mi_cuento_final["continuacion"]))
""")
md("""
### 11. Hallazgos, límites y respuesta al problema
Las cifras siguientes se calculan de la ejecución actual. No se escribieron métricas ficticias ni se
rellenaron conclusiones antes de entrenar. La revisión cualitativa debe atender a continuidad de personajes,
concordancia, progresión de la situación y cierre; las métricas léxicas no capturan estos aspectos.
""")
code("""
selected = results.set_index("modelo").loc[best_name]
bg = results.set_index("modelo").loc["Bigrama"]
improvement = 100*(1-selected.perplexity/bg.perplexity)
gpt_table = validation[validation.modelo.str.startswith("GPT")].set_index("modelo")
depth_message = ("tres capas obtuvieron menor NLL de validación" if gpt_table.loc["GPT-3c", "val_nll"] < gpt_table.loc["GPT-1c", "val_nll"]
                 else "una capa obtuvo menor o igual NLL de validación")
metrics_rag = rag_samples.groupby("recuperacion")[["copia5", "tema_lexico"]].mean()
conclusions = f'''**Resultados medidos**

- La selección por validación eligió **{best_name}**. Su perplejidad de prueba fue **{selected.perplexity:.2f}**
  frente a **{bg.perplexity:.2f}** del bigrama: una reducción relativa de **{improvement:.1f}%**.
  Su intervalo por familias fue **[{selected.ci_low:.2f}, {selected.ci_high:.2f}]**; no incorpora variación entre entrenamientos.
- En la ablación de profundidad, **{depth_message}**. Tener más parámetros no garantiza generalizar mejor;
  esta comparación mantiene épocas y datos, no coste computacional.
- La recuperación obtuvo hit@1 temático **{retrieval_results.hit1.mean():.0%}** en diez consultas manuales.
  La coincidencia de cinco palabras en la continuación pasó de **{metrics_rag.loc[False, "copia5"]:.1%}**
  sin recuperación a **{metrics_rag.loc[True, "copia5"]:.1%}** con recuperación. Son indicadores exploratorios,
  no una evaluación humana de relevancia o creatividad.
- El checkpoint final tiene una coincidencia media de cinco palabras de **{late_samples[late_samples.modelo == "GPT-3c-final"].copia5.mean():.1%}**
  en sus generaciones, frente a **{late_samples[late_samples.modelo == "GPT-3c"].copia5.mean():.1%}** del checkpoint temprano
  de la misma arquitectura. Se debe examinar si ganar fluidez significa reproducir las estructuras conocidas.
- El experimento aprendió patrones de un corpus pequeño, repetitivo y sintético. La separación por familias
  evita compartir la misma plantilla, pero los temas y el estilo fueron diseñados por una misma fuente.
  No es válido extrapolar estos resultados a cuentos de autores humanos ni a un producto editorial.

**Respuesta al problema:** la implementación funciona como laboratorio, pero las muestras del checkpoint seleccionado
presentan palabras deformadas, cambios de personaje y fragmentos sin conexión. **La utilidad como asistente literario
no queda demostrada**. Tener menor NLL que un bigrama y terminar con EOS no resuelve la coherencia narrativa.
El valor de recuperar una frase debe equilibrarse con su copia explícita y la posible continuación memorizada.

**Siguientes pasos:** reunir textos con permiso de uso de varios autores y separar por autor; repetir el entrenamiento
con al menos tres semillas; comparar BPE y un modelo preentrenado ajustado; hacer una evaluación ciega con lectores
que puntúen coherencia, sorpresa y utilidad; comparar recuperación frente a un prefijo humano de igual longitud.
'''
display(Markdown(conclusions))
(ROOT / "artifacts/conclusiones.md").write_text(conclusions)
""")
md("""
### 12. Trazabilidad y reproducibilidad
El manifiesto fija los datos y el vocabulario, configuración, semilla, checkpoint elegido y versiones.
CPU determinista y semillas controlan esta ejecución; entre sistemas, BLAS o versiones de PyTorch puede
haber pequeñas diferencias numéricas. Los tiempos de entrenamiento no son reproducibles como métricas exactas.
No se afirma haber validado todos los sistemas operativos. Las salidas adjuntas corresponden al entorno impreso.
""")
code("""
manifest = {"runtime": runtime, "seed": SEED, "epochs": 24, "configs": CONFIGS,
            "best_validation": best_name, "best_gpt": best_gpt_name,
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [ROOT/"data/familias.json", ROOT/"data/microcuentos.csv", ROOT/"artifacts/tokenizer.json"]}}
(ROOT / "artifacts/manifiesto.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
assert np.isfinite(results.test_nll).all()
assert all(len(set(df[df.split == a].familia) & set(df[df.split == b].familia)) == 0
           for a,b in [("train", "val"), ("train", "test"), ("val", "test")])
a = generate(best_gpt, tok, "mar", seed=123, max_new_tokens=25)
b = generate(best_gpt, tok, "mar", seed=123, max_new_tokens=25)
assert a == b, "La generación debe repetirse con la misma semilla."
print("Verificaciones finales: datos sin fuga de familias, métricas finitas y generación determinista.")
print("Artefactos guardados en", ROOT / "artifacts")
""")
md("""
### Referencias y correspondencia con la rúbrica
- [Material de Sesión 5: RAG y LangChain](https://github.com/Ohtar10/icesi-nlp/tree/main/Sesion5).
- [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762): atención y arquitectura Transformer.
- [Lewis et al., Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401): combinación de recuperación y generación.
- [Holtzman et al., The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751): nucleus sampling.
- [PyTorch: scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

| Criterio | Evidencia localizable |
|---|---|
| Notebook completo | Problema, EDA, código visible, entrenamiento, experimentos y conclusiones, §§1–12 |
| Reproducibilidad | Datos locales, versiones, semillas, ejecución limpia, hashes, §§2–4 y 12 |
| Modelo de generación | GPT causal entrenado, curvas, comparación y salidas, §§5–8 |
| Innovación | Split por familias, BPE, ablación, incertidumbre, copia, recuperación y demo, §§3–10 |

La tabla organiza evidencias; la calificación corresponde al docente. Se declara la asistencia de IA en
la redacción del corpus y en la implementación para que el grupo pueda explicar, revisar y defender cada decisión.
""")
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.9.6"}}
nbf.validate(nb)
nbf.write(nb, ROOT / "Microcuentos_GPT.ipynb")
print(f"Notebook construido: {len(cells)} celdas.")
