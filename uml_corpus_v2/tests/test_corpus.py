import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from corpus import (
    ROOT,
    augment,
    negative_candidates,
    project_ecore,
    read,
    screen,
    signature,
    validate,
)


class CorpusTests(unittest.TestCase):
    def graph(self):
        return {
            "classes": [
                {"name": "Customer", "attributes": []},
                {"name": "Order", "attributes": []},
            ],
            "relations": [
                {
                    "kind": "association",
                    "source": "Customer",
                    "target": "Order",
                    "source_multiplicity": None,
                    "target_multiplicity": "0..*",
                }
            ],
        }

    def test_unknown_multiplicity_preserved(self):
        g = self.graph()
        validate(g)
        self.assertIsNone(g["relations"][0]["source_multiplicity"])

    def test_reverse_duplicate_rejected(self):
        g = self.graph()
        r = copy.deepcopy(g["relations"][0])
        r["source"], r["target"] = r["target"], r["source"]
        r["source_multiplicity"], r["target_multiplicity"] = (
            r["target_multiplicity"],
            r["source_multiplicity"],
        )
        g["relations"].append(r)
        with self.assertRaises(ValueError):
            validate(g)

    def test_no_implicit_opposite(self):
        g, _ = project_ecore(ROOT / "sources/uml_specs/zoo/Automaton_rel1.ecore")
        self.assertEqual(g["relations"][0]["target_multiplicity"], "1..*")
        self.assertIsNone(g["relations"][0]["source_multiplicity"])

    def test_mask_both_canonical_sides(self):
        g = self.graph()
        g["relations"][0]["source_multiplicity"] = "1"
        g, _, _ = screen("Customer links to Order.", g)
        self.assertIsNone(g["relations"][0]["source_multiplicity"])
        self.assertIsNone(g["relations"][0]["target_multiplicity"])

    def test_substring_is_not_identifier(self):
        g = {"classes": [{"name": "Net", "attributes": []}], "relations": []}
        _, flags, _ = screen("A Network has data.", g)
        self.assertTrue(any("class_not_lexically" in f for f in flags))

    def test_unjustified_composition_flagged(self):
        g = self.graph()
        g["relations"][0]["kind"] = "composition"
        _, flags, _ = screen("Customer has zero or more Order objects.", g)
        self.assertIn("composition_not_explicitly_supported", flags)

    def test_negatives_are_unique_non_gold(self):
        g = self.graph()
        ns = negative_candidates(g)
        self.assertEqual(len(ns), 4)
        self.assertEqual(len({signature(g)} | {signature(n["target"]) for n in ns}), 5)
        for n in ns:
            validate(n["target"])

    def test_counterfactual_updates_both_sides(self):
        g = self.graph()
        r = {
            "id": "x",
            "requirement": "Customer has zero or more Order objects.",
            "target": g,
            "negatives": negative_candidates(g),
        }
        views = augment(r)
        cf = next(x for x in views if x.get("variant") == "counterfactual")
        self.assertIn("one or more", cf["requirement"])
        self.assertEqual(cf["target"]["relations"][0]["target_multiplicity"], "1..*")
        self.assertEqual(g["relations"][0]["target_multiplicity"], "0..*")

    def test_rename_updates_endpoints(self):
        g = self.graph()
        r = {
            "id": "x",
            "requirement": "Customer has zero or more Order objects.",
            "target": g,
            "negatives": negative_candidates(g),
        }
        renamed = next(x for x in augment(r) if x.get("variant") == "identifier_rename")
        self.assertIn("ExampleCustomer", renamed["requirement"])
        validate(renamed["target"])

    def test_raw_pool_has_no_fabricated_gold(self):
        self.assertTrue(
            all(r["target"] is None for r in read(ROOT / "data/requirements_pool.jsonl"))
        )


if __name__ == "__main__":
    unittest.main()
