"""Microcuentos: corpus controlado, GPT causal, GRU y recuperación BM25.

Todo funciona en CPU y sin descargas durante la ejecución. El notebook incluye
estas definiciones para que el entrenamiento sea visible y revisable.
"""
# %% imports
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SEED = 17
THEMES = ["memoria", "ciudad", "naturaleza", "tecnologia", "mar"]
SPECIAL = ["<pad>", "<bos>", "<eos>"] + [f"<{t}>" for t in THEMES]
PAD, BOS, EOS = 0, 1, 2

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)

def words(text):
    return re.findall(r"[a-záéíóúüñ]+", text.lower())

# %% corpus
def build_corpus(root=Path(".")):
    """Separar familias ANTES de expandirlas evita fugas entre variantes."""
    families = json.loads((root / "data/familias.json").read_text())
    names = ["Ana", "Luis", "Luz", "Daniel", "Sara", "Tomás", "Elena", "Simón"]
    places = ["la plaza", "el parque", "el pueblo", "la estación"]
    moments = ["medianoche", "mediodía", "las seis", "las diez"]
    rows = []
    rng = np.random.default_rng(SEED)
    for theme in THEMES:
        order = rng.permutation(len(families[theme]))
        split_of = {int(i): ("train" if p < 8 else "val" if p < 10 else "test")
                    for p, i in enumerate(order)}
        for i, template in enumerate(families[theme]):
            for v in range(16):
                text = template.format(nombre=names[v % 8],
                                       lugar=places[(v // 4 + i) % 4],
                                       momento=moments[(v + i) % 4])
                rows.append(dict(id=f"{theme}-{i:02d}-{v:02d}",
                                 familia=f"{theme}-{i:02d}", tema=theme,
                                 split=split_of[i], texto=text,
                                 n_palabras=len(words(text))))
    df = pd.DataFrame(rows)
    assert len(df) == 960 and df.texto.nunique() == len(df)
    assert df.groupby("familia").split.nunique().max() == 1
    (root / "data").mkdir(exist_ok=True)
    df.to_csv(root / "data/microcuentos.csv", index=False)
    return df

def fit_tokenizer(df, root=Path(".")):
    """Byte BPE: aprende uniones solo con train; todo UTF-8 es representable."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=768, special_tokens=SPECIAL,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                                  show_progress=False)
    tok.train_from_iterator(df.loc[df.split == "train", "texto"], trainer)
    (root / "artifacts").mkdir(exist_ok=True)
    tok.save(str(root / "artifacts/tokenizer.json"))
    for text in df.texto:
        assert tok.decode(tok.encode(text).ids) == text
    return tok

def encode_rows(df, tok):
    return [[BOS, tok.token_to_id(f"<{r.tema}>")] + tok.encode(r.texto).ids + [EOS]
            for r in df.itertuples()]

def make_batch(sequences, indices):
    rows = [sequences[int(i)] for i in indices]
    length = max(len(x) for x in rows) - 1
    x = torch.full((len(rows), length), PAD, dtype=torch.long)
    y = torch.full_like(x, PAD)
    for i, ids in enumerate(rows):
        x[i, :len(ids)-1] = torch.tensor(ids[:-1])
        y[i, :len(ids)-1] = torch.tensor(ids[1:])
        y[i, 0] = PAD  # No evaluar la etiqueta: es una condición dada.
    return x, y

# %% architectures
class CausalBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = [z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for z in (q, k, v)]
        # La máscara causal impide consultar el token que se intenta predecir.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.out(a.transpose(1, 2).contiguous().view(b, t, d)))
        return x + self.ff(self.ln2(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab, dim=96, layers=2, heads=4, context=256, dropout=.1):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.Sequential(*[CausalBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        assert x.shape[1] <= self.context
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        return self.head(self.norm(self.blocks(h)))

class GRULM(nn.Module):
    def __init__(self, vocab, dim=128, context=256):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.head(h)

class BigramLM(nn.Module):
    """Baseline de conteos con suavizado; mismo vocabulario y targets."""
    def __init__(self, vocab, sequences, alpha=.1):
        super().__init__()
        self.context = 256
        counts = torch.full((vocab, vocab), alpha)
        for ids in sequences:
            for a, b in zip(ids[1:-1], ids[2:]):
                counts[a, b] += 1
        self.register_buffer("log_probs", (counts / counts.sum(1, keepdim=True)).log())

    def forward(self, x):
        return self.log_probs[x]

# %% training
@torch.inference_mode()
def evaluate(model, sequences, batch_size=32):
    model.eval()
    rows = []
    for start in range(0, len(sequences), batch_size):
        x, y = make_batch(sequences, range(start, min(start+batch_size, len(sequences))))
        losses = F.cross_entropy(model(x).transpose(1, 2), y,
                                 ignore_index=PAD, reduction="none")
        for loss, target in zip(losses, y):
            rows.append({"nll_sum": float(loss.sum()), "tokens": int((target != PAD).sum())})
    result = pd.DataFrame(rows)
    result["nll"] = result.nll_sum / result.tokens
    return result

def global_nll(table):
    return float(table.nll_sum.sum() / table.tokens.sum())

CONFIGS = [
    {"name": "GRU-128", "kind": "gru", "dim": 128, "layers": 1, "lr": .002},
    {"name": "GPT-1c", "kind": "gpt", "dim": 96, "layers": 1, "lr": .002},
    {"name": "GPT-3c", "kind": "gpt", "dim": 96, "layers": 3, "lr": .002},
]

def train_experiments(encoded, vocab, root=Path("."), epochs=24):
    """Siempre entrena desde cero. Test no participa en selección ni parada."""
    all_models, histories, summaries = {}, [], []
    baseline = BigramLM(vocab, encoded["train"])
    all_models["Bigrama"] = baseline
    for config in CONFIGS:
        seed_all()
        if config["kind"] == "gru":
            model = GRULM(vocab, config["dim"])
        else:
            model = TinyGPT(vocab, config["dim"], config["layers"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best, best_state, best_epoch = float("inf"), None, 0
        rng = np.random.default_rng(SEED)
        started = time.perf_counter()
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = total_tokens = 0
            indices = rng.permutation(len(encoded["train"]))
            for start in range(0, len(indices), 32):
                x, y = make_batch(encoded["train"], indices[start:start+32])
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=PAD)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                count = int((y != PAD).sum())
                total_loss += float(loss.detach()) * count
                total_tokens += count
            val = global_nll(evaluate(model, encoded["val"]))
            histories.append(dict(modelo=config["name"], epoca=epoch,
                                  train_nll=total_loss/total_tokens, val_nll=val,
                                  lr=optimizer.param_groups[0]["lr"]))
            if val < best:
                best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
            scheduler.step()
            if epoch == 1 or epoch % 4 == 0:
                print(f'{config["name"]} época {epoch:02d}: train={total_loss/total_tokens:.3f}, val={val:.3f}', flush=True)
        if config["name"] == "GPT-3c":
            # Diagnóstico predefinido de sobreentrenamiento; no candidato al ranking principal.
            late = copy.deepcopy(model).eval()
            all_models["GPT-3c-final"] = late
            summaries.append(dict(modelo="GPT-3c-final", mejor_epoca=epochs,
                                  parametros=sum(p.numel() for p in late.parameters()),
                                  segundos=time.perf_counter()-started, val_nll=val,
                                  train_nll=global_nll(evaluate(late, encoded["train"]))))
            torch.save({"config": config, "state_dict": late.state_dict(), "vocab": vocab},
                       root / "artifacts/GPT-3c-final.pt")
        model.load_state_dict(best_state)
        model.eval()
        all_models[config["name"]] = model
        elapsed = time.perf_counter()-started
        summaries.append(dict(modelo=config["name"], mejor_epoca=best_epoch,
                              parametros=sum(p.numel() for p in model.parameters()), segundos=elapsed,
                              val_nll=best, train_nll=global_nll(evaluate(model, encoded["train"]))))
        torch.save({"config": config, "state_dict": best_state, "vocab": vocab},
                   root / "artifacts" / f'{config["name"]}.pt')
    summaries.append(dict(modelo="Bigrama", mejor_epoca=0, parametros=vocab*vocab,
                          segundos=0., val_nll=global_nll(evaluate(baseline, encoded["val"])),
                          train_nll=global_nll(evaluate(baseline, encoded["train"]))))
    history, summary = pd.DataFrame(histories), pd.DataFrame(summaries)
    history.to_csv(root / "artifacts/entrenamiento.csv", index=False)
    summary.to_csv(root / "artifacts/validacion.csv", index=False)
    return all_models, history, summary

def bootstrap_family(table, seed=SEED, n=1000):
    """Remuestrear familias, no variantes correlacionadas de un mismo relato."""
    grouped = table.groupby("familia")[["nll_sum", "tokens"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    samples = grouped[rng.integers(0, len(grouped), size=(n, len(grouped)))].sum(axis=1)
    return np.quantile(np.exp(samples[:, 0]/samples[:, 1]), [.025, .975])

# %% retrieval
STOP = set("a al de del el la los las un una en que y su sus se por para con no lo como".split())

class BM25:
    def __init__(self, train_df, k1=1.5, b=.75):
        # Una variante por familia: no llenar top-k con casi duplicados.
        self.docs = train_df.drop_duplicates("familia").reset_index(drop=True)
        self.counts = [Counter(w for w in words(t) if w not in STOP) for t in self.docs.texto]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg = self.lengths.mean()
        self.df = Counter(w for c in self.counts for w in c)
        self.k1, self.b = k1, b

    def search(self, query, k=3):
        scores = np.zeros(len(self.docs))
        for w in set(words(query)) - STOP:
            freq = self.df.get(w, 0)
            idf = math.log(1 + (len(self.docs)-freq+.5)/(freq+.5))
            tf = np.array([c.get(w, 0) for c in self.counts])
            scores += idf * tf * (self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths/self.avg))
        order = np.argsort(-scores, kind="stable")[:k]
        return self.docs.iloc[order].assign(score=scores[order]).query("score > 0")

# %% generation
@torch.inference_mode()
def generate(model, tok, theme, prompt="", temperature=.85, top_k=40,
             top_p=.95, seed=17, max_new_tokens=130):
    if theme not in THEMES:
        raise ValueError(f"Tema debe pertenecer a {THEMES}")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("Temperatura > 0, top_p en (0,1], top_k >= 0")
    model.eval()
    ids = [BOS, tok.token_to_id(f"<{theme}>")] + tok.encode(prompt).ids
    if len(ids) + max_new_tokens > model.context:
        raise ValueError("Reduce prompt o max_new_tokens para no exceder el contexto")
    first = len(ids)
    rng = torch.Generator().manual_seed(seed)
    finished = False
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids]))[0, -1] / temperature
        eos_logit = logits[EOS].clone()
        logits[:len(SPECIAL)] = -float("inf")
        # EOS es el único token especial que puede ser generado.
        logits[EOS] = eos_logit if len(ids)-first >= 12 else -float("inf")
        if top_k:
            logits[logits < torch.topk(logits, min(top_k, len(logits))).values[-1]] = -float("inf")
        sorted_logits, indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, -1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[indices[remove]] = -float("inf")
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=rng))
        ids.append(nxt)
        if nxt == EOS:
            finished = True
            break
    continuation = tok.decode(ids[first:])
    return dict(texto=prompt+continuation, continuacion=continuation, eos=finished,
                tokens_nuevos=len(ids)-first)

def ngrams(text, n):
    ws = words(text)
    return [tuple(ws[i:i+n]) for i in range(len(ws)-n+1)]

def generation_metrics(text, train_texts, theme):
    ws = words(text)
    bigrams = ngrams(text, 2)
    lexicon = {
        "memoria": {"recuerdo", "recuerdos", "carta", "infancia", "fotografía", "reloj", "olvidado"},
        "ciudad": {"ciudad", "bus", "calle", "calles", "barrio", "edificios", "ventana"},
        "naturaleza": {"árbol", "semilla", "río", "bosque", "flores", "jardín", "hoja"},
        "tecnologia": {"robot", "máquina", "pantalla", "servidor", "teléfono", "código", "digital"},
        "mar": {"mar", "agua", "barcos", "barco", "faro", "ola", "isla", "océano"},
    }
    reference = set(g for t in train_texts for g in ngrams(t, 5))
    five = ngrams(text, 5)
    candidates = [set(ngrams(t, 3)) for t in train_texts]
    tri = set(ngrams(text, 3))
    nearest = max((len(tri & c)/max(1, len(tri | c)) for c in candidates), default=0.)
    return dict(palabras=len(ws), distinct2=len(set(bigrams))/max(1, len(bigrams)),
                copia5=sum(g in reference for g in five)/max(1, len(five)),
                jaccard_max=nearest, tema_lexico=float(bool(set(ws) & lexicon[theme])))

def demo(model, tok, retriever, tema="mar", idea="una carta junto al mar", recuperar=True,
         temperature=.85, seed=17):
    docs = retriever.search(idea, k=1) if recuperar else pd.DataFrame()
    # Adaptación de RAG a un LM de continuación: prefijo literario, no instrucciones.
    prefix = ""
    if len(docs):
        prefix = docs.iloc[0].texto.split(".", 1)[0] + "."
    out = generate(model, tok, tema, prefix, temperature=temperature, seed=seed)
    out["prefijo_recuperado"] = prefix
    out["fuentes"] = docs[["id", "familia", "score"]].to_dict("records") if len(docs) else []
    return out
