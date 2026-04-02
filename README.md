# Ticket Threads Retrieval

End-to-end **semantic retrieval pipeline for IT help desk ticket threads**.

This project converts resolved TeamsDynamix-style support tickets into supervised training pairs, fine-tunes a `SentenceTransformer`, builds a normalized embedding index, and retrieves the most relevant historical resolutions for new incoming tickets.

The pipeline demonstrates how to bootstrap a practical retrieval system from ticket threads using:

* **synthetic supervision**
* **hard negative sampling**
* **embedding fine-tuning**
* **retrieval evaluation (Recall@K + MRR)**
* optional **grounded answer drafting with Ollama**

---

## Key Results

### Retrieval Performance

Evaluated on **104 unique ticket issues**:

* **Recall@1:** 0.6731
* **Recall@2:** 0.9135
* **Recall@3:** 0.9808
* **Recall@4:** 1.0000
* **Recall@5:** 1.0000
* **MRR:** 0.8205

This shows the correct **resolution category is usually retrieved in the top few results**, with perfect recall by top-4.

### Embedding Validation

Fine-tuned from `all-MiniLM-L6-v2`:

* **Cosine Pearson:** 0.9532
* **Cosine Spearman:** 0.8664
* **Epochs:** 3
* **Batch size:** 16
* **Learning rate:** `2e-5`

These results indicate strong semantic clustering of paraphrased ticket issues and realistic separation from hard negatives.

---

## Baseline vs Fine-Tuned

| Model                     | Recall@1 | Recall@3 |    MRR |
| ------------------------- | -------: | -------: | -----: |
| `all-MiniLM-L6-v2`        |   0.6731 |   0.9712 | 0.8171 |
| Fine-tuned `ticket-pairs` |   0.6731 |   0.9808 | 0.8205 |

The pretrained MiniLM baseline was already strong on IT help desk phrasing, so fine-tuning produced **modest but consistent gains in Recall@2–5 and MRR** rather than dramatic top-1 jumps.

### Why the gain wasn't bigger

The improvement was intentionally realistic and limited by three factors:

* **Strong baseline model:** MiniLM already handles common IT paraphrases well (account lockouts, VPN access, shared drive permissions, printer issues)
* **Small fine-tuning dataset:** only ~200 synthetic tickets were used, which is enough for domain nudging but not large embedding-space shifts
* **Repetitive issue families:** many tickets share templated language, so the pretrained model already clusters them effectively

This was still a valuable ML finding:

> the primary bottleneck shifted from embedding retrieval quality to the **richness of resolution knowledge units**

In other words, the model usually retrieved the correct **resolution family**, but downstream usefulness is now more constrained by how detailed and actionable the stored fixes are.

---

## Problem This Solves

IT support systems often contain thousands of historical tickets, but finding the right prior resolution is difficult because:

* users phrase the same issue differently
* keyword search misses paraphrases
* ticket closures are noisy
* multiple valid fixes may exist for the same issue family

This project solves that by learning embeddings that map:

> **new issue phrasing → historical resolution family**

instead of relying on exact keyword overlap.

---

## Pipeline Overview

The full workflow:

1. Generate or ingest TeamsDynamix-style ticket threads
2. Extract issue-resolution pairs from resolved tickets with Ollama
3. Create supervised positive + hard-negative pairs
4. Fine-tune a `SentenceTransformer`
5. Build normalized embedding retrieval index
6. Retrieve top historical fixes for new tickets
7. Optionally draft a grounded answer with Ollama

---

## Models & Tools

### Embeddings

* Base: `sentence-transformers/all-MiniLM-L6-v2`
* Fine-tuned: `models/ticket-pairs`

### LLM Extraction / Drafting

* Local Ollama API
* Example model: `qwen2.5:7b-instruct`

### Core Libraries

* `sentence-transformers`
* `torch`
* `numpy`
* `requests`

---

## Main Scripts

* `tdx_simulate_tickets.py` → generate synthetic TeamsDynamix-style tickets
* `tdx_ollama_pair_builder.py` → extract issue/resolution pairs
* `train_sentence_transformer.py` → fine-tune embeddings
* `build_ticket_index.py` → build normalized retrieval index
* `answer_new_tickets.py` → retrieve similar resolutions and draft responses
* `evaluate_ticket_retrieval.py` → compute Recall@K and MRR
* `evaluate_extraction.py` → evaluate extraction precision / recall / F1

---

## Example Data Artifacts

Included sample artifacts:

* `simulated_200.jsonl`
* `out_pairs/extracted_pairs.jsonl`
* `out_pairs/train_pairs.csv`
* `out_pairs/triplets.jsonl`
* `ticket_eval_report.json`

---

## Quick Start

### 1) Generate synthetic tickets

```powershell
python tdx_simulate_tickets.py --count 200 --output simulated_200.jsonl
```

### 2) Extract resolved ticket pairs

```powershell
python tdx_ollama_pair_builder.py `
  --input simulated_200.jsonl `
  --output-dir out_pairs `
  --ollama-model qwen2.5:7b-instruct `
  --min-confidence 0.65 `
  --require-closed
```

### 3) Train retrieval embeddings

```powershell
python train_sentence_transformer.py `
  --input out_pairs/train_pairs.csv `
  --output-dir models/ticket-pairs
```

### 4) Build retrieval index

```powershell
python build_ticket_index.py `
  --input out_pairs/extracted_pairs.jsonl `
  --model models/ticket-pairs `
  --output-dir retrieval_index
```

### 5) Evaluate retrieval

```powershell
python evaluate_ticket_retrieval.py `
  --input out_pairs/train_pairs.csv `
  --model models/ticket-pairs `
  --top-k 5 `
  --output ticket_eval_report.json
```

### 6) Answer new tickets

```powershell
python answer_new_tickets.py `
  --input example.json `
  --model models/ticket-pairs `
  --index-dir retrieval_index `
  --output answers.jsonl `
  --top-k 3 `
  --min-score 0.55
```

---

## Retrieval Method

The system uses **L2-normalized sentence embeddings**.

Similarity scoring:

```python
score = query_embedding @ corpus_embedding
```

Because both vectors are normalized, this dot product is equivalent to:

> **cosine similarity**

This makes retrieval fast and numerically stable.

---

## Key ML Insight

A major finding from this project:

> **retrieval quality became stronger than answer usefulness**

The system reliably retrieved the correct resolution family for paraphrased tickets, but practical usefulness was limited by the brevity and generic phrasing of stored resolutions rather than by embedding retrieval quality.


My next major improvement step is richer resolution synthesis.

---

## Recommended GitHub Structure

Commit:

* source scripts
* README
* synthetic sample data
* processed pair data
* evaluation reports

Exclude:

* `.venv/`
* `models/`
* `retrieval_index/`
* checkpoints
* `.npy`
* large generated outputs

---

## Future Improvements

- Improve how resolutions are stored by generating richer fix summaries that include the root cause, steps taken, verification, and follow-up recommendations
- Add category-level confusion matrix analysis to better understand which issue types are most often confused during retrieval
- Evaluate on a true held-out train/test split to measure generalization on unseen ticket issues
- Combine BM25 keyword search with dense embeddings for better handling of exact IT terms, software names, and error codes
- Build a lightweight UI demo for help desk agents to test ticket retrieval interactively
- Add sample screenshots showing example ticket inputs, retrieved historical fixes, and confidence scores
