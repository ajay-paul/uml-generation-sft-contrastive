# Generating UML from natural language with SFT and contrastive learning

Research code and saved outputs for Ajay Paul's bachelor thesis, **Generating Diagrams from Natural Language Requirements Using Small Language Models**, at Hamm-Lippstadt University of Applied Sciences.

The experiment compares **Llama 3.1 8B Instruct** and **Qwen3 8B** in three conditions: the original checkpoint without a task adapter, supervised fine tuning (SFT), and SFT combined with a supervised sequence-level contrastive loss. The reported evaluation contains **ten original requirements shared by all six configurations**, producing 60 completions. The prepared test partition has 55 records; this repository's reported results concern the selected ten.

## Repository structure

```text
uml_corpus_v2/                 Source data, preprocessing, grouped splits and tests
  src/                        Fetching, projection, screening and augmentation
  sources/                    Source payloads, licences, metadata and checksums
  data/                       Provisional pairs, quarantine and five prepared folds
  annotations/                Review material
uml_source_run_fold0/          The prepared experiment (run commands from here)
  src/                        Training, generation, scoring, plotting and chat tools
  scripts/                    Experiment and ten-item evaluation commands
  data/                       Exact prepared data used by the saved experiment
  models.lock.json            Recorded model checkpoint revisions
  environment*.json/txt       Recorded VM environment (not newly measured)
  results_test10_run1/         Saved 60 completions, references, scores and charts
  runs/                       Created by training; model weights are not distributed
 docs/training_runs/           Original run histories and adapter configurations
 docs/REPOSITORY_CLEANUP.md    Packaging changes and verification scope
```

The two original folder names are retained to preserve script paths. Dataset preparation copies the experiment's current code into a fresh workspace; a separate stale `runtime/` code copy is not maintained.

## Research questions

1. Does task-specific fine tuning improve structural validity and reference agreement over the original checkpoints?
2. Does the additional contrastive objective improve generation over SFT alone?
3. Which errors remain, and how do the dataset and evaluation limitations affect interpretation?

## Data and provenance

The labelled source is the English–UML benchmark associated with Yang and Sahraoui, *Towards Automatically Extracting UML Class Diagrams from Natural Language Specifications* ([publication](https://doi.org/10.1145/3550356.3561592), [dataset](https://github.com/songyang-dev/uml-classes-and-specs)). Volunteers described existing UML fragments. These examples should not be described as naturally collected industrial requirements.

This preprocessing pipeline screened 649 candidate pairs, retained 271 provisional pairs and quarantined 378 with recorded flags. These are this project's preparation counts, not results claimed by the source paper. Automated checks do not establish independent expert validation. Fold 0 contains 162 original training pairs, 54 validation pairs and 55 test pairs. Training-only augmentation produces 526 positive rows. The prepared file contains 635 rows in total: 526 training, 54 validation and 55 test rows.

PURE passages and published user stories remain an exploratory, unlabelled pool. They were not used for training, validation or testing. Review drafts are explicitly marked and are not approved labels. See [the corpus documentation](uml_corpus_v2/README.md) and [third-party notices](uml_corpus_v2/THIRD_PARTY_NOTICES.md) for source attribution and terms. Original source licences and their third-party caveats are retained; this repository does not apply a blanket new licence to the mixed dataset contents.

## Training method

Each model has two independently trained adapters. Both start from the original checkpoint; the combined run does not continue from the saved SFT adapter.

For requirement `x` and JSON completion `y`, the sequence score is the average log probability of its completion tokens. SFT minimizes the negative score of the reference. Combined training additionally applies a softmax cross-entropy over the reference and two selected negative completions, with the reference as the correct candidate. Four negatives are stored per positive row; two are selected with deterministic rotation across epochs. This is conditional sequence scoring, not cosine similarity or embedding-distance training.

The configured loss is `SFT loss + 0.1 × contrastive loss`, with contrastive temperature `0.1`. Training uses eight epochs, seed 42, LoRA rank 16, alpha 32, dropout 0, 4-bit NF4 weights, BF16 computation, learning rate `1e-4`, and accumulation over eight requirement rows. The checkpoint with the lowest validation reference-completion loss is retained. These are tested settings, not established optimal settings.

