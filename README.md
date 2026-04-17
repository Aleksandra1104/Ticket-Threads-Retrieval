# Ticket Threads Retrieval

End-to-end semantic retrieval pipeline for IT help desk ticket threads.

This repo now has two clear layers:

- reusable Python packages for simulation and extraction under `ticket_memory/`
- runnable scripts grouped by task under `extraction/`, `training/`, `indexing/`, `retrieval/`, `evaluation/`, and `viewers/`

The core question behind the project is still:

> Can historical ticket threads be converted into reusable retrieval memory for future support incidents?

## Pipeline

The workflow is designed as a **modular simulation → extraction → retrieval → evaluation system**, where each stage can be improved independently.

1. **Generate reusable synthetic ticket threads**  
   Use the modular `support_sim/` simulator with reusable dataclasses, domain rules, configurable personas, and multi-step support flows (`direct_resolution`, escalation, partial resolution, mixed issues). Hidden structured ground truth is generated for later evaluation.

2. **Extract issue-resolution pairs from raw ticket threads**  
   Convert noisy multi-turn threads into structured retrieval memories using blind Ollama extraction plus strict JSON schema validation, category whitelists, confidence thresholds, deterministic post-validation rules, and skipped-ticket auditing.

3. **Build supervised training pairs**  
   Create positive issue → resolution pairs and optional hard negatives / triplets for dense retrieval training.

4. **Fine-tune `SentenceTransformer`**  
   Train embeddings so semantically similar support incidents cluster together even when phrased differently.

5. **Build normalized retrieval index**  
   Encode extracted issue memories into an L2-normalized dense vector store for fast cosine-similarity retrieval.

6. **Retrieve top historical fixes for new tickets**  
   Match new issues against historical resolutions using top-k semantic nearest-neighbor retrieval.

7. **Evaluate extraction + retrieval quality**  
   Validate both upstream data generation and downstream retrieval using:
   - extraction precision / recall / F1
   - category and family accuracy
   - text similarity metrics
   - confusion matrices
   - `Recall@K`
   - `MRR`

8. **Review tickets and retrieval matches in Streamlit**  
   Inspect simulated threads, extracted pairs, skipped tickets, retrieval scores, and failure cases interactively.

9. **Optionally draft grounded support responses with Ollama**  
   Use retrieved historical fixes as grounding context for suggested agent replies and future RAG-style help desk workflows.



## Repo Layout

```text
ticket_memory/
|-- simulation/
|   |-- core/
|   |-- domains/
|   |-- exporters/
|   `-- examples/
`-- extraction/
    |-- base.py
    |-- models.py
    |-- ollama_extractor.py
    |-- pipeline.py
    `-- thread_render.py

extraction/
`-- extract_ticket_pairs.py

training/
`-- train_sentence_transformer.py

indexing/
`-- build_ticket_index.py

retrieval/
`-- answer_new_tickets.py

evaluation/
|-- evaluate_extraction.py
`-- evaluate_ticket_retrieval.py

viewers/
|-- streamlit_ticket_viewer.py
`-- streamlit_retrieval_viewer.py
```

### `ticket_memory/simulation/`

The modular simulator package.

- `core/`: domain-agnostic models, engine, flows, personas, rendering helpers
- `domains/it_support/`: IT issue catalog, artifacts, and rules
- `exporters/`: raw thread and retrieval-pair exporters
- `examples/generate_it_threads.py`: clean example generator entrypoint

The simulator supports:

- reusable dataclasses such as `Scenario`, `Message`, and `IssueVariant`
- configurable flow types including direct resolution, escalation, partial resolution, failed first fix, and mixed issues
- user personas such as vague, frustrated, technical, and cooperative
- agent personas such as concise, methodical, and empathetic
- optional secondary issues and resolution states

### `ticket_memory/extraction/`

The modular extraction package.

- `thread_render.py`: converts structured tickets into extraction-ready thread text
- `ollama_extractor.py`: Ollama-backed thread-only extraction
- `pipeline.py`: validation and extraction orchestration
- `models.py` and `base.py`: extraction-side interfaces and dataclasses

This layer is designed to treat simulated threads like real help desk data, without relying on hidden simulator labels for the extracted pair itself.

### Grouped script folders

These folders contain the runnable CLIs and review tools:

- `extraction/`: build extracted pairs, train pairs, triplets, and skipped-ticket logs
- `training/`: train sentence-transformer models
- `indexing/`: build dense retrieval indexes
- `retrieval/`: retrieve similar historical fixes and optionally draft grounded replies
- `evaluation/`: measure extraction quality and retrieval quality
- `viewers/`: Streamlit apps for human review

The repo root still keeps thin compatibility wrappers such as [extract_ticket_pairs.py](/C:/ticket_threads_retrieval/extract_ticket_pairs.py), [build_ticket_index.py](/C:/ticket_threads_retrieval/build_ticket_index.py), and [streamlit_ticket_viewer.py](/C:/ticket_threads_retrieval/streamlit_ticket_viewer.py) so older commands continue to work.

## IT Taxonomy

The current IT support taxonomy includes:

- IAM: `account_locked`, `password_reset`, `mfa_issue`, `permission_issue`, `onboarding_offboarding`
- Networking: `vpn_issue`, `wifi_connectivity`, `internet_access`, `voip_telephony`
- Hardware: `workstation_failure`, `peripheral_issue`, `printer_issue`, `mobile_device_issue`
- Software: `email_issue`, `software_install`, `application_crash`, `browser_issue`
- Storage: `shared_drive_issue`, `data_recovery`, `disk_space_full`
- Security: `phishing_report`, `malware_infection`, `encryption_issue`
- Backend: `server_unavailable`, `database_connection`, `api_failure`
- Fallback: `other`

## Main Entry Points

### Simulation

- [tdx_simulate_tickets.py](/C:/ticket_threads_retrieval/tdx_simulate_tickets.py): original single-file simulator
- [ticket_memory/simulation/examples/generate_it_threads.py](/C:/ticket_threads_retrieval/ticket_memory/simulation/examples/generate_it_threads.py): modular simulator example

### Extraction

- tdx_ollama_pair_builder.py: original end-to-end extraction script
- extraction/extract_ticket_pairs.py: cleaner grouped extraction CLI

### Training

- training/train_sentence_transformer.py

### Indexing

- indexing/build_ticket_index.py

### Retrieval

- retrieval/answer_new_tickets.py

### Evaluation

- evaluation/evaluate_extraction.py
- evaluation/evaluate_ticket_retrieval.py

### Streamlit Review Tools

- viewers/streamlit_ticket_viewer.py: browse simulated tickets and full threads
- viewers/streamlit_retrieval_viewer.py: inspect top-k retrieval matches for a new ticket

## Quick Start

### 1. Generate synthetic tickets

```powershell
python ticket_memory/simulation/examples/generate_it_threads.py `
  --count 200 `
  --output simulated_modular_200.jsonl `
  --pairs-output out_pairs/simulated_modular_pairs.jsonl
```

### 2. Extract ticket pairs

```powershell
python extraction/extract_ticket_pairs.py `
  --input simulated_modular_200.jsonl `
  --output-dir out_pairs_modular `
  --ollama-model qwen2.5:7b-instruct `
  --min-confidence 0.65 `
  --require-closed
```

### 3. Train embeddings

```powershell
python training/train_sentence_transformer.py `
  --input out_pairs_modular/train_pairs.csv `
  --output-dir models/ticket-pairs
```
### Validation strategy update

During pair fine-tuning, validation originally used `CosineSimilarityEvaluator`, which is designed for continuous semantic similarity tasks where labels represent graded similarity scores.

Because this project uses **binary relevance labels** for support ticket pairs:

- `1` → matching issue / correct historical resolution
- `0` → non-matching pair / hard negative

validation was updated to use `BinaryClassificationEvaluator`.

This evaluator:

1. computes similarity scores on held-out validation pairs
2. finds the best similarity threshold
3. converts similarities into match / non-match predictions
4. selects the best checkpoint using validation **accuracy**

This better matches the downstream retrieval objective:

> distinguish relevant historical tickets from irrelevant ones

The final training setup is:

- **Loss:** `CosineSimilarityLoss`
- **Validation:** `BinaryClassificationEvaluator`
- **System evaluation:** `Recall@K`, `MRR`

This hybrid design keeps embedding learning smooth while selecting the checkpoint that best separates relevant from irrelevant support incidents.

### 4. Build a retrieval index

```powershell
python indexing/build_ticket_index.py `
  --input out_pairs_modular/extracted_pairs.jsonl `
  --model models/ticket-pairs `
  --output-dir retrieval_index
```

### 5. Evaluate extraction

```powershell
python evaluation/evaluate_extraction.py `
  --tickets simulated_modular_200.jsonl `
  --extracted out_pairs_modular/extracted_pairs.jsonl `
  --skipped out_pairs_modular/skipped_tickets.jsonl `
  --output-dir extraction_eval `
  --only-resolved
```

### 6. Evaluate retrieval

```powershell
python evaluation/evaluate_ticket_retrieval.py `
  --input out_pairs_modular/train_pairs.csv `
  --model models/ticket-pairs `
  --top-k 5 `
  --relevance-mode exact `
  --output ticket_eval_report.json
```
```powershell
python evaluation/evaluate_ticket_retrieval.py `
  --input out_pairs_modular/train_pairs.csv `
  --model models/ticket-pairs `
  --top-k 5 `
  --relevance-mode category `
  --output ticket_eval_category.json
```

### 7. Retrieve similar historical fixes for a new ticket

```powershell
python retrieval/answer_new_tickets.py `
  --input example.json `
  --model models/ticket-pairs `
  --index-dir retrieval_index `
  --output answers.jsonl `
  --top-k 3 `
  --min-score 0.55
```

### 8. Review tickets in Streamlit

```powershell
python -m streamlit run viewers/streamlit_ticket_viewer.py
```

### 9. Review retrieval matches in Streamlit

```powershell
python -m streamlit run viewers/streamlit_retrieval_viewer.py
```

## Evaluation Notes

`evaluation/evaluate_extraction.py` reports:

- extraction precision, recall, and F1
- fine-grained category accuracy
- family-level accuracy
- issue and resolution text-similarity metrics
- confusion matrices
- skipped-reason breakdowns

`evaluation/evaluate_ticket_retrieval.py` reports:

- `Recall@K`
- `MRR`
- top-1 exact-category accuracy
- top-1 family accuracy
- mean first-positive rank
- median first-positive rank

## Current Results

### Extraction quality

- Precision: `1.0000`
- Recall: `0.9767`
- F1: `0.9882`
- Fine-grained category accuracy: `0.9286`
- Family accuracy: `0.9524`

### Retrieval quality

Evaluated on 130 unique ticket issues using different relevance definitions:

#### Exact match (same ticket)

- `Recall@1`: `0.215`
- `Recall@3`: `0.562`
- `Recall@5`: `0.754`
- `MRR`: `0.437`
- median first-positive rank: `3`

This measures how often the model retrieves the *exact same historical ticket*.

The relatively low top-1 accuracy is expected: many tickets share **equivalent resolutions expressed with different wording or slight variations** (e.g., MFA resync vs re-enrollment, printer reinstall vs remapping). The model often retrieves a *semantically correct alternative* rather than the exact original instance.

#### Category-level match (same issue type)

- `Recall@1`: `0.915`
- `Recall@3`: `0.923`
- `Recall@5`: `0.977`
- `MRR`: `0.934`

This reflects practical usefulness: retrieving a fix from the correct issue category.

In most cases, even when the exact ticket is not retrieved, the model returns a resolution with the correct remediation pattern (e.g., MFA reset, printer reinstall, VPN profile refresh).

#### Summary

- The model is **moderately accurate at exact ticket retrieval**
- The model is **highly accurate at retrieving the correct type of resolution**
- For over 90% of queries, the top result already belongs to the correct issue category

This indicates the embedding space clusters tickets by **resolution semantics rather than exact instance identity**, which is desirable for support-assist workflows.


## Stack

- `sentence-transformers`
- `torch`
- `numpy`
- `requests`
- `streamlit`
- Ollama with `qwen2.5:7b-instruct`

## Notes

- The root-level scripts remain as compatibility wrappers so existing commands do not break while the grouped layout becomes the default.
