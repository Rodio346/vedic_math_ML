"""CLI: attention DAG structural comparison (Flash vs Vedic tiling)."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from attention_dag.analysis.compare import (
    compare_strategies,
    compare_strategies_analytical,
)

CSV_COLUMNS = [
    "seq_len",
    "d_head",
    "tile_r",
    "tile_c",
    "tile_d",
    "flash_critical_path",
    "vedic_critical_path",
    "flash_per_tile_cp",
    "vedic_per_embed_tile_cp",
    "concurrent_tiles_flash",
    "concurrent_tiles_vedic",
    "inter_tile_dependency_flash",
    "inter_tile_dependency_vedic",
    "efficiency_ratio",
    "flash_parallel_width",
    "vedic_parallel_width",
    "flash_total_ops",
    "vedic_total_ops",
    "dag_global_flash_cp",
    "dag_global_vedic_cp",
    "crossover_workers",
    "structural_verdict",
]


def _worker_counts(seq_len: int, d_head: int) -> list[int | float]:
    raw = [
        1,
        d_head // 4,
        d_head // 2,
        d_head,
        2 * d_head,
        seq_len,
        seq_len * d_head,
        math.inf,
    ]
    counts: list[int | float] = []
    seen: set[int | float] = set()
    for w in raw:
        if w < 1 and w != math.inf:
            continue
        if w not in seen:
            seen.add(w)
            counts.append(w)
    return counts


def _ensure_tests_pass(attention_dir: Path, *, skip: bool) -> None:
    if skip:
        return
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=attention_dir,
        check=False,
    )
    if result.returncode != 0:
        print(
            "Refusing to run: pytest tests/ failed. Fix tests before main.py.",
            file=sys.stderr,
        )
        sys.exit(1)


def _write_summary(
    summary_path: Path,
    rows: list[dict[str, object]],
) -> None:
    by_pair: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_pair[(int(row["seq_len"]), int(row["d_head"]))].append(row)

    lines: list[str] = []
    lines.append("Attention DAG structural summary")
    lines.append("=" * 60)
    lines.append(
        "Fair unit: per-tile CP along d_head (= D for both). Verdict uses "
        "concurrent_tiles (Vedic D/tile_d vs Flash 1), not global DAG CP."
    )

    for (seq_len, d_head), group in sorted(by_pair.items()):
        lines.append("")
        lines.append(f"seq_len={seq_len}, d_head={d_head}")
        lines.append("-" * 40)

        vedic_wins = [r for r in group if r["structural_verdict"] == "vedic_shorter"]
        flash_wins = [r for r in group if r["structural_verdict"] == "flash_shorter"]
        identical = [r for r in group if r["structural_verdict"] == "identical"]

        sample = group[0]
        lines.append(
            f"Per-tile CP: flash={sample['flash_per_tile_cp']} "
            f"vedic_embed_tile={sample['vedic_per_embed_tile_cp']} "
            f"(full d_head reduction = {sample['vedic_critical_path']})."
        )
        lines.append(
            f"Inter-tile: flash={sample['inter_tile_dependency_flash']}, "
            f"vedic={sample['inter_tile_dependency_vedic']}."
        )

        if vedic_wins:
            best = max(vedic_wins, key=lambda r: float(r["efficiency_ratio"]))
            lines.append(
                f"Vedic higher concurrent embed tiles in {len(vedic_wins)}/{len(group)} "
                f"settings (best concurrent_tiles_vedic={best['concurrent_tiles_vedic']}, "
                f"efficiency_ratio={best['efficiency_ratio']}, tile_d={best['tile_d']})."
            )
            lines.append(
                "Recommendation: structural parallelism advantage warrants "
                "Component 2 (memory analysis)."
            )
        elif flash_wins:
            lines.append(
                f"Flash higher concurrent tiles in {len(flash_wins)}/{len(group)} settings."
            )
            lines.append(
                "Recommendation: do not proceed to Component 2 on structural grounds alone."
            )
        else:
            lines.append(
                f"Identical concurrent_tiles in all {len(identical)} settings "
                f"(tile_d >= d_head); per-tile CP still equal."
            )
            lines.append(
                "Recommendation: no embed-tile parallelism gain; arc complete unless "
                "memory analysis finds another angle."
            )

        if group[0].get("dag_global_flash_cp"):
            lines.append(
                f"Reference global DAG CP (unfair row vs score unit): "
                f"flash={group[0]['dag_global_flash_cp']} "
                f"vedic={group[0]['dag_global_vedic_cp']}."
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    seq_lens: list[int],
    d_heads: list[int],
    tile_r: int,
    tile_c: int,
    tile_ds: list[int],
    output_path: Path,
    *,
    skip_test_gate: bool = False,
    analytical: bool = False,
) -> None:
    attention_dir = Path(__file__).resolve().parent
    _ensure_tests_pass(attention_dir, skip=skip_test_gate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.parent / "attention_dag_summary.txt"
    all_rows: list[dict[str, object]] = []

    with output_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for seq_len in seq_lens:
            for d_head in d_heads:
                for tile_d in tile_ds:
                    if analytical:
                        result = compare_strategies_analytical(
                            seq_len, d_head, tile_r, tile_c, tile_d
                        )
                    else:
                        workers = _worker_counts(seq_len, d_head)
                        result = compare_strategies(
                            seq_len,
                            d_head,
                            tile_r,
                            tile_c,
                            tile_d,
                            workers,
                        )
                    row = {
                        "seq_len": seq_len,
                        "d_head": d_head,
                        "tile_r": tile_r,
                        "tile_c": tile_c,
                        "tile_d": tile_d,
                        "flash_critical_path": result.flash_critical_path,
                        "vedic_critical_path": result.vedic_critical_path,
                        "flash_per_tile_cp": result.flash_per_tile_cp,
                        "vedic_per_embed_tile_cp": result.vedic_per_embed_tile_cp,
                        "concurrent_tiles_flash": result.concurrent_tiles_flash,
                        "concurrent_tiles_vedic": result.concurrent_tiles_vedic,
                        "inter_tile_dependency_flash": result.inter_tile_dependency_flash,
                        "inter_tile_dependency_vedic": result.inter_tile_dependency_vedic,
                        "efficiency_ratio": round(result.efficiency_ratio, 6),
                        "flash_parallel_width": result.flash_parallel_width,
                        "vedic_parallel_width": result.vedic_parallel_width,
                        "flash_total_ops": result.flash_total_ops,
                        "vedic_total_ops": result.vedic_total_ops,
                        "dag_global_flash_cp": result.dag_global_flash_cp or "",
                        "dag_global_vedic_cp": result.dag_global_vedic_cp or "",
                        "crossover_workers": result.crossover_workers
                        if result.crossover_workers is not None
                        else "",
                        "structural_verdict": result.structural_verdict,
                    }
                    writer.writerow(row)
                    all_rows.append(row)

    _write_summary(summary_path, all_rows)
    print(f"Wrote {output_path} ({len(all_rows)} configurations)")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attention score DAG structural comparison (Flash vs Vedic).",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    parser.add_argument(
        "--d-heads",
        type=int,
        nargs="+",
        default=[32, 64, 128],
    )
    parser.add_argument("--tile-r", type=int, default=16)
    parser.add_argument("--tile-c", type=int, default=16)
    parser.add_argument(
        "--tile-d",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "attention_dag.csv",
    )
    parser.add_argument(
        "--skip-test-gate",
        action="store_true",
        help="Skip pytest gate (development only).",
    )
    parser.add_argument(
        "--analytical",
        action="store_true",
        help="Use closed-form tile-unit metrics (no DAG build; for large seq_len).",
    )
    args = parser.parse_args()

    if min(args.seq_lens + args.d_heads + [args.tile_r, args.tile_c] + args.tile_d) < 1:
        print("All dimensions must be >= 1", file=sys.stderr)
        sys.exit(1)

    run_analysis(
        args.seq_lens,
        args.d_heads,
        args.tile_r,
        args.tile_c,
        args.tile_d,
        args.output,
        skip_test_gate=args.skip_test_gate,
        analytical=args.analytical,
    )


if __name__ == "__main__":
    main()
