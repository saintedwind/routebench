"""Pinned, crowdsourced CLINC150 records and the authors' domain mapping."""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
UPSTREAM = f"https://raw.githubusercontent.com/clinc/oos-eval/{REVISION}/"
FILES = {
    "data/data_full.json": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    "data/domains.json": "b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a",
    "LICENSE": "e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76",
}
OUTSIDE = "out_of_scope"
SAMPLE_SALT = "routebench-clinc-test-v1"
SPLIT_COUNTS = {"train": 15000, "val": 3000, "test": 4500, "oos_val": 100, "oos_test": 1000}


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str


def verify(payload: bytes, name: str) -> bytes:
    if hashlib.sha256(payload).hexdigest() != FILES[name]:
        raise ValueError(f"SHA-256 mismatch for {name}")
    return payload


def fetch(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = directory / Path(name).name
        if target.exists():
            verify(target.read_bytes(), name)
            continue
        with urlopen(UPSTREAM + name, timeout=40) as response:
            payload = verify(response.read(4_000_001), name)
        with NamedTemporaryFile(dir=directory, delete=False) as handle:
            temp = Path(handle.name)
            try:
                handle.write(payload)
                handle.close()
                temp.replace(target)
            finally:
                handle.close()
                temp.unlink(missing_ok=True)


def load(directory: Path) -> dict[str, list[Example]]:
    def read(name):
        return json.loads(verify((directory / Path(name).name).read_bytes(), name))

    source = read("data/data_full.json")
    domains = read("data/domains.json")
    label_to_domain = {intent: domain for domain, intents in domains.items() for intent in intents}
    if len(domains) != 10 or len(label_to_domain) != 150:
        raise ValueError("Unexpected domain mapping")
    result = {}
    for split, count in SPLIT_COUNTS.items():
        if len(source[split]) != count:
            raise ValueError(f"Unexpected number of examples in {split}")
        rows = []
        for index, (text, label) in enumerate(source[split]):
            if split.startswith("oos_") and label != "oos":
                raise ValueError(f"Unexpected out-of-scope label in {split}")
            label = OUTSIDE if label == "oos" else label_to_domain[label]
            rows.append(Example(f"{split}:{index}", text, label))
        result[split] = rows
    return result


def sample_test(examples: list[Example], per_domain: int = 20, outside: int = 40) -> list[Example]:
    """Fixed hash order within each label, independent of system predictions."""
    grouped = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)
    selected = []
    for label in sorted(grouped):
        limit = outside if label == OUTSIDE else per_domain
        ordered = sorted(
            grouped[label], key=lambda e: hashlib.sha256(f"{SAMPLE_SALT}:{e.id}".encode()).digest()
        )
        if not 1 <= limit <= len(ordered):
            raise ValueError(f"Invalid sample size for {label}")
        selected.extend(ordered[:limit])
    return sorted(selected, key=lambda e: e.id)
