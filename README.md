#  KGRAG — Knowledge-Graph based RAG


---

## Quick start

# Testing Command:
$ python tests/pipeline.py [--input /path/to/docs_or_dir] [--config cfg.yaml]
Or:
$ sbatch run_pipeline.sh (or modify the script to use srun if preferred)

# Output:
1. Save an extracted graph to the `results/` directory
2. Print a short summary

# 1  Create / activate Conda env (Python ≥3.9)
```bash
conda create -n kgrag python=3.9 -y
conda activate kgrag
pip install -r requirements.txt
```

# 2  Download / cache models (login node with internet)
```bash
python src/models/download_models.py   # optional – or place models manually in $HF_HUB_CACHE
```

# 3  Run locally
```bash
python tests/pipeline.py                # built-in sample docs
python tests/pipeline.py --input docs/  # your own folder
```
Results are written to `results/` (JSON < 1 k nodes, Parquet otherwise).

# 4  Run on the cluster
```bash
sbatch run.sh           # SLURM script (GPU-aware)
```
Edit `run.sh` to tweak partition, time, or cache paths.

---

##  Repository layout
```
configs/              YAML configuration (model names, chunk sizes…)
results/              Saved graphs (JSON / Parquet)
src/
 ├─ data/             – document loading + chunking
 ├─ kg/               – entity extraction, merging, storage helpers
 └─ models/           – model manager, prompt templates
tests/
 ├─ pipeline.py       – concise end-to-end pipeline script
 ├─ test_data.py      – small sample documents
 └─ …
run.sh                – SLURM submission script
```

---


##  TODO
- Improve relationship merging logic.
- Large-scale evaluation & retrieval benchmarking.
- Experiment with implicit-entity discovery from transformer embeddings.


--


## Potential Further Research:
- Extract implicit entities from transformer embeddings.
    
    Key Research Questions:
    1. Can we find conceptually relevant but unmentioned entities?
    2. How do we validate that latent entities are meaningful?
    3. What's the optimal sampling strategy from embedding space?

    Methods:
    1. embedding_neighbors: Find similar entities in embedding space
    2. attention_analysis: Use attention weights to find important tokens
    3. masking_inference: Mask text and see what model expects
    4. compositional_reasoning: Combine explicit entities conceptually

- Reducing edge complexity and improving traversal efficiency through hierarchical organization and shared property abstraction.

    Key Ideas:
    1. Create shared property nodes (e.g., "mammal_properties")
    2. Build taxonomic hierarchies  
    3. Compress redundant edges
    4. Enable efficient multi-level traversal

- Learn some random walking policy for graph traversal?