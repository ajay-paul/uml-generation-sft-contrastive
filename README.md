# UML Generation with SFT and Contrastive Learning

Generate UML class-model fragments from natural-language requirements using
**Llama 3.1 8B Instruct** and **Qwen3 8B**.

## Methods

The study compares:

- Original models without task-specific adaptation
- Supervised fine-tuning (SFT)
- SFT with sequence-level contrastive learning

Training uses LoRA adapters with 4-bit quantized model weights. Outputs are
JSON graphs containing classes, attributes, relationships and multiplicities.

## Evaluation

Evaluation includes structural validity, fact-based precision, recall and F1,
and exact normalized graph matching.

On a supplementary sample of ten held-out requirements, both adaptation
methods improved performance over the original models. Contrastive learning
provided no additional mean generation improvement over SFT alone.

## Dataset

The study uses paired English–UML data from Yang and Sahraoui (2022):
*Towards Automatically Extracting UML Class Diagrams from Natural Language
Specifications*.

Reference graphs remain provisional after automated screening.

## Scope

This research prototype generates class-model fragments, not complete UML
models. Agreement with a reference graph does not guarantee that every aspect
of a requirement has been represented.
