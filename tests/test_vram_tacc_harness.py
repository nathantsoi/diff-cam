import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts" / "vram_scaling_matrix.tsv"
SLURM = ROOT / "scripts" / "vram_scaling.slurm"


def _rows():
    with MATRIX.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _dense_gib(row):
    n = int(row["N"])
    steps = int(row["T"])
    voxels = n**3
    return (8 * (steps + 1) * voxels + 24 * voxels) / 1024**3


def test_vram_matrix_has_contiguous_unique_probe_ids():
    rows = _rows()
    assert [int(row["probe_id"]) for row in rows] == list(range(31))
    assert all(row["method"] in {"delta", "sweep"} for row in rows)


def test_vram_matrix_covers_both_scaling_axes_and_a100_walls():
    rows = _rows()
    full_rows = rows[1:]

    assert {row["T"] for row in full_rows if row["sweep_axis"] == "T_at_N128"} == {
        "64",
        "128",
        "256",
        "1024",
        "2048",
        "2560",
        "5120",
    }
    assert {row["N"] for row in full_rows if row["sweep_axis"] == "N_at_T128"} == {
        "48",
        "96",
        "128",
        "192",
        "256",
        "320",
        "352",
        "448",
    }

    by_id = {int(row["probe_id"]): row for row in rows}
    assert 39.5 < _dense_gib(by_id[11]) < 41.0
    assert 79.5 < _dense_gib(by_id[13]) < 81.0
    assert _dense_gib(by_id[27]) > 40.0
    assert _dense_gib(by_id[29]) > 80.0


def test_lonestar6_script_avoids_unsupported_gpu_directives():
    script = SLURM.read_text(encoding="utf-8")
    assert "#SBATCH -p gpu-a100-small" in script
    assert "#SBATCH --array=0-2%3" in script
    assert "--gres" not in script
    assert "--gpus-per-task" not in script
