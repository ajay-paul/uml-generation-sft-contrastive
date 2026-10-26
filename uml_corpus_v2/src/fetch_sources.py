"""Restore pinned public source payloads; verify hashes before extracting selected files."""

import argparse
import csv
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

from corpus import ROOT


def download(url, path, algorithm, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.new(algorithm, data).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + url)
        path.write_bytes(data)
    elif hashlib.new(algorithm, path.read_bytes()).hexdigest() != expected:
        raise ValueError("Local checksum mismatch: " + str(path))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-documents", action="store_true")
    a = p.parse_args()
    for collection in ["pure", "stories"]:
        metadata = json.loads((ROOT / "sources" / collection / "metadata.json").read_text())
        for f in metadata["files"]:
            if f["key"] == "requirements.zip" and not a.original_documents:
                continue
            algo, checksum = f["checksum"].split(":", 1)
            path = ROOT / "sources" / collection / f["key"]
            download(f["links"]["self"], path, algo, checksum)
    lock = json.loads((ROOT / "sources/downloads.lock.json").read_text())
    asset = next(x for x in lock if x["id"] == "uml_specs_release")
    archive = download(asset["url"], ROOT / "cache/dataset.tar.gz", "sha256", asset["sha256"])
    dest = ROOT / "sources/uml_specs"
    with tarfile.open(archive) as t:
        for name in ["labels.csv", "fragments.csv", "models.csv"]:
            (dest / name).write_bytes(t.extractfile(name).read())
        labels = list(csv.DictReader((dest / "labels.csv").open()))
        fragments = {r["unique_id"]: r for r in csv.DictReader((dest / "fragments.csv").open())}
        (dest / "zoo").mkdir(exist_ok=True)
        for row in labels:
            f = fragments[row["fragment_id"]]
            stem = f"{f['model']}_{f['kind']}{f['number']}"
            if Path(stem).name != stem:
                raise ValueError("Unsafe archive member name")
            for ext in [".ecore", ".plantuml"]:
                (dest / "zoo" / (stem + ext)).write_bytes(t.extractfile("zoo/" + stem + ext).read())
    print("Source payloads verified/restored. Originals stay in their collection folders.")


if __name__ == "__main__":
    main()
