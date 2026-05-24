# Attention DAG — Component 1 Structural Analysis

This component tests whether **Urdhva-Tiryagbhyam-style embed-dimension tiling** offers a structural scheduling advantage over **FlashAttention-style sequence tiling** for transformer attention score computation (`QK^T`). It is a pure DAG exercise: no PyTorch, no GPU kernels, no real tensors—only dependency graphs, the Phase 2B discrete-time scheduler, and crossover analysis.

## What this tests and why

Phase 2B compared multiplication algorithms by operand-specific DAGs. Here we hold the **operation count fixed** (same `S²×D` multiplications and `S²×(D−1)` additions) and vary only **dependency order** to see if Vedic column-sequencing along `d_head` yields a shorter critical path or better parallelism than FlashAttention’s `seq_len` tiling. A positive structural result motivates **Component 2** (memory / IO analysis); a negative result stops before that work.

## DAG structures

### Shared per-score invariant

For every score position `(i, j)`:

- **D** independent `MULT` nodes (`Q[i,d] * K[j,d]`)
- **D−1** `ADD` nodes in a **serial accumulation chain** (depth `D−1`)

Global: `S²×D` MULT, `S²×(D−1)` ADD. If these counts differ between builders, there is a bug.

### Vedic embed tiling (`vedic_attn_dag.py`)

Parameters: `seq_len=S`, `d_head=D`, `tile_d=T`.

```
For each embed tile t (d = 0..D-1 step T):
  For ALL (i,j) simultaneously:
    T MULTs (independent)
    T-1 ADDs chained within tile
    Tile boundary (t>0): first ADD of tile t also depends on
      final ADD of tile t-1 AND MULT of tile t, d_local=0
```

ASCII (`D=4`, `T=2`, one score):

```
  [M0,M1]--ADD0--||--[M2]--ADD1--ADD2
   tile0          ^      tile1 (boundary + chain)
                  boundary link to ADD1
```

**Critical path per score (unlimited workers, this DAG):**

`CP_vedic = 2*T` when `D = 2*T` (two full embed tiles): one parallel MULT wave, then `T` serial ADD stages per tile after the first.

Example: `D=4`, `T=2` → CP = **4**.

Idealized bound `D/T + T - 1` (e.g. **3** for `D=4`, `T=2`) assumes tile-boundary and within-tile adds collapse into fewer scheduler steps than this explicit `D-1` ADD chain model.

At `D=64`, `tile_d=8`: per-score CP = **16** vs Flash row CP below.

### FlashAttention seq tiling (`flashattn_dag.py`)

Parameters: `seq_len=S`, `d_head=D`, `tile_r=Br`, `tile_c=Bc`.

```
For each query tile / key tile block (i,j):
  D MULTs (independent)
  D-1 ADDs chained over full D for this (i,j)
Key-tile boundary: first ADD of scores in the next key tile for
  query row i depends on final ADD of previous key tile’s last
  column for row i (online softmax accumulator model).
```

ASCII (`D=4`, one score; row chain across `S/Bc` key tiles):

```
  M0..M3 --ADD0--ADD1--ADD2     (depth on D)
  score[i,j0] ----> score[i,j1]  (S/Bc - 1 row steps)
```

**Critical path per query row (unlimited workers, this DAG):**

`CP_flash = 2*(D-1) + (S/Bc - 1)`

Each key tile after the first adds another full `(D-1)`-deep ADD chain on the row, because the online-softmax boundary feeds the previous tile’s last column into the first ADD of every score in the next tile.

Example: `S=4`, `D=4`, `Bc=2` → CP = **7**.

Idealized bound `1 + (D-1) + (S/Bc - 1)` (e.g. **5**) counts only one embed reduction plus key-tile boundaries, not a repeated full `D` chain per key tile.

## Fair comparison unit (tile-unit metrics)

Comparisons use the **same output unit**: one tile of score computations along `d_head`.

| Metric | Flash | Vedic |
|--------|-------|-------|
| Per-tile CP (one Br×Bc score tile or one embed-tile wave) | **D** | **tile_d** per embed round; **D** total along head |
| `concurrent_tiles` | **1** (key tiles along seq are sequential) | **D / tile_d** (embed tiles are independent partial sums) |
| `inter_tile_dependency` | **sequential** (online softmax) | **none** |

`flash_critical_path` and `vedic_critical_path` in the CSV are both **D** (full dot-product depth per score). They are equal by construction; do not use them for the verdict.

**`structural_verdict`** uses **`concurrent_tiles_vedic` vs `concurrent_tiles_flash`**:

| Verdict | Meaning | Next step |
|---------|---------|-----------|
| `vedic_shorter` | Vedic has more concurrent embed-tile rounds (`D/tile_d` &gt; 1) | Warrant **Component 2** (memory analysis) |
| `identical` | `tile_d >= d_head` (only one embed round) | No embed-tile parallelism gain |
| `flash_shorter` | Rare (only if concurrent tiles favor Flash) | Do not proceed to Component 2 on structure alone |

`efficiency_ratio = concurrent_tiles_vedic / concurrent_tiles_flash` (parallelism ratio, not CP ratio).

`dag_global_flash_cp` / `dag_global_vedic_cp` are reference full-graph scheduler depths (unfair row-vs-score units); kept for debugging only.

`crossover_workers`: smallest `W > 1` in the sweep where Vedic completion time is **strictly less** than Flash’s.

## Requirements

- Python 3.10+
- `pytest` (repo `requirements-dev.txt`)
- Run from `attention_dag/` (or repo root with path on `PYTHONPATH`)

## How to run

**Tests first (required):**

```powershell
cd attention_dag
python -m pytest tests/ -v
```

All six structure tests and two parity tests must pass, including analytical CP formula tests.

**Smoke run:**

```powershell
python main.py --seq-lens 4 8 --d-heads 4 8 --tile-r 2 --tile-c 2 --tile-d 2 --output results/smoke.csv
Get-Content results/attention_dag_summary.txt
```

**Full analysis:**

```powershell
python main.py --seq-lens 64 128 256 512 --d-heads 32 64 128 --tile-r 16 --tile-c 16 --tile-d 8 16 32 --output results/attention_dag.csv
```

`main.py` refuses to run until `pytest tests/` passes (use `--skip-test-gate` only for development).

## Layout

```
attention_dag/
├── dag/
│   ├── node.py              # copied from phase2b
│   ├── scheduler.py           # copied from phase2b
│   ├── flashattn_dag.py
│   └── vedic_attn_dag.py
├── analysis/
│   └── compare.py
├── tests/
├── results/
├── main.py
└── README.md
```
