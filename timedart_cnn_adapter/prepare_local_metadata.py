"""Create private recording identities and CV folds locally from a user-provided ZIP."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from data import canonical

ROOT = Path(__file__).resolve().parent

def build_metadata(data_path, seed=42):
    records = []
    seen_names, seen_hashes = set(), set()
    reference_time = None
    with zipfile.ZipFile(data_path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".dat") and not name.startswith("__MACOSX/")
        )
        for name in names:
            identifier = canonical(name)
            if identifier in seen_names:
                raise ValueError("Duplicate canonical recording name")
            seen_names.add(identifier)
            array = np.loadtxt(io.BytesIO(archive.read(name)))
            if array.shape == (50001, 2):
                array = array.T
            if array.shape != (2, 50001) or not np.isfinite(array).all():
                raise ValueError("Expected a finite two-column/row 50,001-point DAT recording")
            time, voltage = array
            if not np.all(np.diff(time) > 0):
                raise ValueError("Nonmonotonic time axis")
            if reference_time is None:
                reference_time = time
            elif not np.allclose(time, reference_time, rtol=0, atol=1e-15):
                raise ValueError("Recording time axes differ")
            digest = hashlib.sha256(voltage.astype(np.float32).tobytes()).hexdigest()
            if digest in seen_hashes:
                raise ValueError("Duplicate waveform content")
            seen_hashes.add(digest)
            records.append({
                "file": identifier,
                "label": int("##결함/" in identifier),
                "waveform_sha256": digest,
            })
    records.sort(key=lambda row: row["file"])
    labels = np.asarray([row["label"] for row in records], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if len(records) != 144 or counts.tolist() != [126, 18]:
        raise ValueError("This experiment requires 126 normal and 18 defect recordings")
    indices = np.arange(len(records))
    documents = []
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(indices, labels), 1):
        inner_seed = 1000 + fold
        train, validation = train_test_split(
            development, test_size=0.20, random_state=inner_seed,
            stratify=labels[development],
        )
        roles = {"train": train, "validation": validation, "test": test}
        sets = {role: set(ids.tolist()) for role, ids in roles.items()}
        if (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"]):
            raise RuntimeError("Split overlap")
        documents.append({
            "fold": fold,
            "outer_seed": seed,
            "inner_seed": inner_seed,
            "unit": "original recording",
            "stratification": "label only",
            "splits": {
                role: [dict(records[int(i)]) for i in ids]
                for role, ids in roles.items()
            },
        })
    test_ids = [row["file"] for doc in documents for row in doc["splits"]["test"]]
    if len(test_ids) != 144 or len(set(test_ids)) != 144:
        raise RuntimeError("Every recording must appear in outer test exactly once")
    dataset = {"counts": {"normal": 126, "defect": 18}, "records": records}
    return dataset, documents

def prepare(data_path, seed=42):
    dataset, folds = build_metadata(data_path, seed)
    outputs = {
        "dataset_manifest.json": dataset,
        "fixed_folds.json": folds,
    }
    existing = [name for name in outputs if (ROOT / name).exists()]
    if existing:
        if len(existing) != len(outputs):
            raise ValueError("Incomplete local metadata; use a fresh project directory")
        for name, content in outputs.items():
            saved = json.loads((ROOT / name).read_text(encoding="utf-8"))
            if saved != content:
                raise ValueError("Existing local metadata differs; use a fresh project directory")
        print("Existing local metadata matches this dataset and split seed.")
        return
    for name, content in outputs.items():
        path = ROOT / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    print("Private recording metadata and five folds generated locally.")
    print("The public repository contains no recording identity lists.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    prepare(arguments.data, arguments.seed)
