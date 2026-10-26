import hashlib
import json
import re
from pathlib import Path

from corpus import canonical
from corpus import validate as validate

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def relation_tuple(r, full=False):
    s, t = norm(r["source"]), norm(r["target"])
    sm, tm = r["source_multiplicity"], r["target_multiplicity"]
    if r["kind"] == "association" and s > t:
        s, t, sm, tm = t, s, tm, sm
    return (r["kind"], s, t, sm, tm) if full else (r["kind"], s, t)


def facts(g, category):
    if category == "classes":
        return {(norm(c["name"]),) for c in g["classes"]}
    if category == "attributes":
        return {
            (norm(c["name"]), norm(a["name"]), a["type"])
            for c in g["classes"]
            for a in c["attributes"]
        }
    if category in {"relations", "full_relations"}:
        return {relation_tuple(r, category == "full_relations") for r in g["relations"]}
    if category == "known_multiplicities":
        result = set()
        for r in g["relations"]:
            kind, source, target, sm, tm = relation_tuple(r, True)
            if kind == "generalization":
                continue
            if sm is not None:
                result.add((kind, source, target, "source", sm))
            if tm is not None:
                result.add((kind, source, target, "target", tm))
        return result
    if category == "graph":
        return {(k,) + x for k in ["classes", "attributes", "full_relations"] for x in facts(g, k)}
    raise ValueError(category)


def serialize(g):
    return json.dumps(canonical(g), ensure_ascii=False, separators=(",", ":"))


SYSTEM = """Convert the specification into one JSON object, without Markdown or explanation.
Schema: {"classes":[{"name":"ClassName","attributes":[{"name":"attributeName","type":null}]}],"relations":[{"kind":"association","source":"ClassA","target":"ClassB","source_multiplicity":null,"target_multiplicity":"0..*"}]}.
Represent only the stated class-model fragment. Do not complete it from outside knowledge.
Retain class and attribute identifiers; use PascalCase for multiword class names and lowerCamelCase for multiword attribute names when necessary.
Attribute type is its explicit type name, or null if unspecified. Multiplicity is an explicit bound such as 1, 0..1, 0..*, 1..*, 2..5, or null if unspecified. Unknown is not zero and not a default multiplicity.
source_multiplicity counts source objects per target; target_multiplicity counts target objects per source. Do not invent the opposite multiplicity from a one-way statement.
Kinds: association, composition, generalization. Composition goes from whole to part; generalization from subclass to superclass with null multiplicities. Associations are undirected; do not emit reversed duplicates.
This task projects away role names, navigation arrows, abstract flags, operations, attribute multiplicities and ordering. Include only directly stated attributes. Do not invent ownership from vague containment language. No additional keys."""


def split_ids(fold=0, protocol="within"):
    manifest = json.loads((ROOT / "data/corpus_manifest.json").read_text())
    if protocol != "within" or fold != manifest["active_fold"]:
        raise ValueError(
            "Prepare a separate corpus workspace for the requested fold; within now means source-family held-out."
        )
    rows = read_jsonl(ROOT / "data/augmented.jsonl")
    return {r["parent_id"]: ("val" if r["split"] == "validation" else r["split"]) for r in rows}


def select_rows(split, fold=0, protocol="within", views="all"):
    groups = split_ids(fold, protocol)
    rows = [r for r in read_jsonl(ROOT / "data/augmented.jsonl") if groups[r["parent_id"]] == split]
    if views == "original":
        rows = [r for r in rows if r["variant"] == "original"]
    return rows
