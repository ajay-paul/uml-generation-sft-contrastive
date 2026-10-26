"""Dependency-free provenance, graph projection and contrastive augmentation helpers."""

import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XSI = "{http://www.w3.org/2001/XMLSchema-instance}type"


def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def relation_key(r):
    s, t, sm, tm = r["source"], r["target"], r["source_multiplicity"], r["target_multiplicity"]
    if r["kind"] == "association" and s > t:
        s, t, sm, tm = t, s, tm, sm
    return (r["kind"], s, t, sm, tm)


def canonical(g):
    g = copy.deepcopy(g)
    g["classes"].sort(key=lambda c: c["name"])
    for c in g["classes"]:
        c["attributes"].sort(key=lambda a: a["name"])
    for r in g["relations"]:
        k = relation_key(r)
        r.update(
            kind=k[0], source=k[1], target=k[2], source_multiplicity=k[3], target_multiplicity=k[4]
        )
    g["relations"].sort(key=lambda r: json.dumps(relation_key(r)))
    return g


def signature(g):
    return sha(json.dumps(canonical(g), sort_keys=True))


def validate(g):
    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    require(isinstance(g, dict) and set(g) == {"classes", "relations"}, "graph keys")
    require(isinstance(g["classes"], list) and 1 <= len(g["classes"]) <= 20, "class count")
    require(isinstance(g["relations"], list) and len(g["relations"]) <= 60, "relation count")
    names = []
    for c in g["classes"]:
        require(isinstance(c, dict) and set(c) == {"name", "attributes"}, "class keys")
        require(
            isinstance(c["name"], str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", c["name"]),
            "class identifier",
        )
        names.append(c["name"])
        attrs = []
        require(isinstance(c["attributes"], list), "attributes list")
        for a in c["attributes"]:
            require(isinstance(a, dict) and set(a) == {"name", "type"}, "attribute keys")
            require(
                isinstance(a["name"], str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", a["name"]),
                "attribute identifier",
            )
            require(
                a["type"] is None
                or isinstance(a["type"], str)
                and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", a["type"]),
                "attribute type",
            )
            attrs.append(norm(a["name"]))
        require(len(attrs) == len(set(attrs)), "duplicate attribute")
    require(len(names) == len(set(map(norm, names))), "duplicate class")
    seen = set()
    edges = {"composition": {}, "generalization": {}}
    for r in g["relations"]:
        require(
            isinstance(r, dict)
            and set(r)
            == {"kind", "source", "target", "source_multiplicity", "target_multiplicity"},
            "relation keys",
        )
        require(r["kind"] in {"association", "composition", "generalization"}, "relation kind")
        require(
            isinstance(r["source"], str)
            and isinstance(r["target"], str)
            and r["source"] in names
            and r["target"] in names,
            "endpoints",
        )
        for k in ["source_multiplicity", "target_multiplicity"]:
            m = r[k]
            require(
                m is None or isinstance(m, str) and re.fullmatch(r"\d+(?:\.\.(?:\d+|\*))?", m),
                "multiplicity syntax",
            )
            if isinstance(m, str) and ".." in m:
                lo, hi = m.split("..")
                require(hi == "*" or int(lo) <= int(hi), "multiplicity bounds")
        if r["kind"] == "generalization":
            require(
                r["source_multiplicity"] is None and r["target_multiplicity"] is None,
                "inheritance multiplicity",
            )
        if r["kind"] == "composition":
            require(
                r["source_multiplicity"] in {None, "1", "0..1"}, "composition owner multiplicity"
            )
        key = relation_key(r)
        require(key not in seen, "duplicate relation")
        seen.add(key)
        if r["kind"] in edges:
            edges[r["kind"]].setdefault(r["source"], []).append(r["target"])
    for links in edges.values():

        def visit(n, stack):
            require(n not in stack, "hierarchy cycle")
            for nxt in links.get(n, []):
                visit(nxt, stack | {n})

        for n in links:
            visit(n, set())
    return True


def project_ecore(path):
    root = ET.parse(path).getroot()
    nodes = [
        e
        for e in root.iter()
        if e.tag.split("}")[-1] == "eClassifiers" and e.get(XSI) == "ecore:EClass"
    ]
    g = {"classes": [], "relations": []}
    omissions = set()
    for node in nodes:
        c = {"name": node.attrib["name"], "attributes": []}
        g["classes"].append(c)
        if node.get("abstract") == "true":
            omissions.add("abstract_flag")
        for parent in node.get("eSuperTypes", "").split():
            g["relations"].append(
                {
                    "kind": "generalization",
                    "source": c["name"],
                    "target": parent.rsplit("/", 1)[-1],
                    "source_multiplicity": None,
                    "target_multiplicity": None,
                }
            )
        for f in node:
            kind = f.get(XSI, "")
            if kind == "ecore:EAttribute":
                typ = f.get("eType")
                if typ is None:
                    et = next((x for x in f if x.tag.split("}")[-1] == "eType"), None)
                    typ = et.get("href") if et is not None else None
                typ = typ.rsplit("/", 1)[-1] if typ else None
                c["attributes"].append({"name": f.attrib["name"], "type": typ})
                omissions.add("attribute_multiplicity_and_order")
            elif kind == "ecore:EReference":
                target = f.get("eType", "").rsplit("/", 1)[-1]
                lo = f.get("lowerBound", "0")
                hi = f.get("upperBound", "1")
                hi = "*" if hi == "-1" else hi
                mult = lo if lo == hi else lo + ".." + hi
                g["relations"].append(
                    {
                        "kind": "composition" if f.get("containment") == "true" else "association",
                        "source": c["name"],
                        "target": target,
                        "source_multiplicity": None,
                        "target_multiplicity": mult,
                    }
                )
                omissions.update(["role_names", "navigability", "reference_order"])
    # EReference bounds constrain targets per source. An absent opposite is UNKNOWN, never 1.
    unique = {relation_key(r): r for r in g["relations"]}
    g["relations"] = list(unique.values())
    validate(g)
    return canonical(g), sorted(omissions)


def screen(text, graph):
    """Conservative lexical screening, not semantic verification. """
    issues = []
    target = copy.deepcopy(graph)
    masked = []
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    spans = {norm("".join(tokens[i : i + n])) for i in range(len(tokens)) for n in range(1, 7)}

    def grounded(name):
        value = norm(name)
        return value in spans or value + "s" in spans or value + "es" in spans

    for c in target["classes"]:
        if not grounded(c["name"]):
            issues.append("class_not_lexically_grounded:" + c["name"])
        for a in c["attributes"]:
            if not grounded(a["name"]):
                issues.append("attribute_not_lexically_grounded:" + c["name"] + "." + a["name"])
            if a["type"] and not re.search(r"\b" + re.escape(a["type"]) + r"\b", text, re.I):
                masked.append("attribute_type:" + c["name"] + "." + a["name"])
                a["type"] = None
    # Catch an explicit attribute list item not represented by the reference fragment.
    known = {norm(a["name"]) for c in graph["classes"] for a in c["attributes"]}
    for name in re.findall(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+(?:as|of type)\s+(?:String|Integer|Boolean|Double|Float|DateTime)\b",
        text,
        re.I,
    ):
        if norm(name) not in known:
            issues.append("text_attribute_absent_from_reference:" + name)
    cardinality = re.search(
        r"\b(?:zero|one|two|many|several|single|multiple|exactly|at least|at most|any number)\b|\d+\.\.(?:\d+|\*)",
        text,
        re.I,
    )
    for i, r in enumerate(target["relations"]):
        if r["kind"] != "generalization" and not cardinality:
            for field in ["source_multiplicity", "target_multiplicity"]:
                if r[field] is not None:
                    r[field] = None
                    masked.append(field + ":" + str(i))
        if r["kind"] == "composition" and not re.search(
            r"compos|exclusiv|life.?cycle|lifetime|destroy|delet", text, re.I
        ):
            issues.append("composition_not_explicitly_supported")
        if r["kind"] == "association" and re.search(
            r"\b(?:owned|composition|composite|exclusively)\b", text, re.I
        ):
            issues.append("ownership_language_needs_review")
    if len(target["relations"]) == 1:
        known = [
            target["relations"][0][k]
            for k in ["source_multiplicity", "target_multiplicity"]
            if target["relations"][0][k] is not None
        ]
        for phrase, mult in [("zero or more", "0..*"), ("one or more", "1..*")]:
            if phrase in text.lower() and len(known) == 1 and known[0] != mult:
                issues.append("text_multiplicity_conflicts_with_reference")
    if (
        len(graph["classes"]) == 1
        and not graph["classes"][0]["attributes"]
        and not graph["relations"]
    ):
        issues.append("class_only_without_structural_detail")
    if len(text.split()) < 4:
        issues.append("insufficient_text")
    return target, sorted(set(issues)), masked


def negative_candidates(g):
    """One localized mutation when possible; valid, unique and different from gold."""
    result = []
    seen = {signature(g)}

    def add(h, kind, reason, difficulty="structural"):
        try:
            validate(h)
        except (ValueError, TypeError):
            return
        sig = signature(h)
        if sig in seen:
            return
        seen.add(sig)
        result.append(
            {"mutation": kind, "reason": reason, "difficulty": difficulty, "target": canonical(h)}
        )

    for i, r in enumerate(g["relations"]):
        if r["kind"] != "generalization":
            for field in ["target_multiplicity", "source_multiplicity"]:
                old = r[field]
                if old is not None:
                    h = copy.deepcopy(g)
                    h["relations"][i][field] = "1" if old != "1" else "0..*"
                    add(
                        h,
                        "multiplicity",
                        f"Changes {field} from {old} to {h['relations'][i][field]}.",
                    )
        h = copy.deepcopy(g)
        h["relations"].pop(i)
        add(h, "missing_relation", "Removes a reference relation.")
        if r["kind"] == "generalization":
            h = copy.deepcopy(g)
            h["relations"][i]["source"], h["relations"][i]["target"] = r["target"], r["source"]
            add(h, "inheritance_direction", "Reverses subclass and superclass.")
        else:
            h = copy.deepcopy(g)
            h["relations"][i]["kind"] = (
                "association" if r["kind"] == "composition" else "composition"
            )
            add(h, "relation_kind", "Changes association/ownership semantics.")
            h = copy.deepcopy(g)
            h["relations"][i]["source"], h["relations"][i]["target"] = r["target"], r["source"]
            add(
                h,
                "endpoint_assignment",
                "Reassigns ownership or the constrained multiplicity to the opposite endpoint.",
            )
    for i, c in enumerate(g["classes"]):
        for j, a in enumerate(c["attributes"]):
            h = copy.deepcopy(g)
            h["classes"][i]["attributes"].pop(j)
            add(h, "missing_attribute", f"Removes {c['name']}.{a['name']}.")
            if a["type"] is not None:
                h = copy.deepcopy(g)
                h["classes"][i]["attributes"][j]["type"] = (
                    "Boolean" if a["type"] != "Boolean" else "String"
                )
                add(h, "attribute_type", "Replaces the explicit attribute type.")
            for other in range(len(g["classes"])):
                if other == i:
                    continue
                h = copy.deepcopy(g)
                att = h["classes"][i]["attributes"].pop(j)
                h["classes"][other]["attributes"].append(att)
                add(h, "attribute_owner", "Moves an attribute to a different class.")
    for i, c in enumerate(g["classes"]):
        for name in ["identifier", "description", "status", "createdAt"]:
            if name in {a["name"] for a in c["attributes"]}:
                continue
            h = copy.deepcopy(g)
            h["classes"][i]["attributes"].append({"name": name, "type": None})
            add(
                h,
                "unsupported_attribute",
                f"Adds unannotated attribute {c['name']}.{name}; valid only under the closed annotated projection.",
            )
    # Fallbacks are explicitly labelled EASY, never called hard negatives.
    for name in ["Account", "Document", "Record", "User", "Service", "Event"]:
        if name in {c["name"] for c in g["classes"]}:
            continue
        h = copy.deepcopy(g)
        old = h["classes"][0]["name"]
        h["classes"][0]["name"] = name
        for r in h["relations"]:
            for k in ["source", "target"]:
                if r[k] == old:
                    r[k] = name
        add(h, "class_identity", f"Replaces required class {old} with {name}.", "easy")
    return result[:4]


def augment(row):
    """Two bounded task wrappers; never mutate quantifiers, names or sentence order."""
    result = [copy.deepcopy(row)]
    wrappers = [
        ("instruction_prefix", "Represent the following specification as a class diagram.\n\n"),
        ("quoted_specification", "The specification to model is given below.\n\n"),
    ]
    for i, (kind, prefix) in enumerate(wrappers, 1):
        r = copy.deepcopy(row)
        r["id"] = row["id"] + f"-A{i}"
        r["variant"] = kind
        r["requirement"] = prefix + row["requirement"]
        r["augmentation"] = {
            "method": kind,
            "preserves_original_text": True,
            "generator": "deterministic-v1",
        }
        result.append(r)
    names = [c["name"] for c in row["target"]["classes"]]
    attrs = {a["name"] for c in row["target"]["classes"] for a in c["attributes"]}
    if not attrs.intersection(names) and all(
        re.search(r"\b" + re.escape(n) + r"\b", row["requirement"]) for n in names
    ):
        mapping = {n: "Example" + n for n in names}
        if all(len(n) <= 64 for n in mapping.values()):
            r = copy.deepcopy(row)
            r["id"] = row["id"] + "-AR"
            r["variant"] = "identifier_rename"
            pattern = (
                r"\b(?:"
                + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                + r")\b"
            )
            r["requirement"] = re.sub(pattern, lambda m: mapping[m.group()], r["requirement"])
            for c in r["target"]["classes"]:
                c["name"] = mapping[c["name"]]
            for rel in r["target"]["relations"]:
                rel["source"] = mapping[rel["source"]]
                rel["target"] = mapping[rel["target"]]
            r["augmentation"] = {
                "method": "consistent_identifier_rename",
                "mapping": mapping,
                "generator": "deterministic-v1",
            }
            validate(r["target"])
            r["negatives"] = negative_candidates(r["target"])
            result.append(r)
    relations = row["target"]["relations"]
    if len(relations) == 1 and relations[0]["target_multiplicity"] in {"0..*", "1..*"}:
        old = relations[0]["target_multiplicity"]
        new = "1..*" if old == "0..*" else "0..*"
        choices = [("zero or more", "one or more"), ("0..*", "1..*")]
        if old == "1..*":
            choices = [(b, a) for a, b in choices]
        for before, after in choices:
            if row["requirement"].count(before) != 1:
                continue
            r = copy.deepcopy(row)
            r["id"] = row["id"] + "-CF"
            r["variant"] = "counterfactual"
            r["requirement"] = row["requirement"].replace(before, after)
            r["target"]["relations"][0]["target_multiplicity"] = new
            r["augmentation"] = {
                "method": "paired_cardinality_change",
                "before": before,
                "after": after,
                "generator": "deterministic-v1",
            }
            validate(r["target"])
            r["negatives"] = negative_candidates(r["target"])
            result.append(r)
            break
    return result
