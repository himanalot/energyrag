# E-RAG: Energy-Based Retrieval-Augmented Generation

A prototype system that unifies document retrieval and sequence generation under a single “energy” objective, with a live Gradio GUI for comparing standard RAG vs. E-RAG on your own documents (PDFs or text).

---

## 🚀 Why It Matters

- **Grounded Generation**  
  Traditional RAG pipelines retrieve then generate in two isolated steps, which can lead to hallucinations. E-RAG trains an energy function over *(context, answer)* pairs so the generator learns to stay faithful to its evidence and the retriever learns which passages truly help the model answer.

- **Contrastive Learning**  
  By explicitly contrasting good vs. bad context–answer pairs, E-RAG discriminates fine-grained differences, reducing spurious matches and boosting answer precision—critical when accuracy and trustworthiness matter (e.g. medical, legal, scientific domains).

- **Built-In Confidence Scoring**  
  The learned energy score acts as a natural confidence measure: lower energy → higher consistency between retrieved text and generated output. This can drive intelligent fallback, re-ranked suggestions, or human-in-the-loop review.

- **Interactive, Demo-Ready GUI**  
  Our Gradio Blocks interface lets you upload any PDF or text file, ask questions, and watch standard RAG vs. E-RAG answers (and their latencies) appear in sequence—ideal for rapid prototyping, user studies, or stakeholder demos.

---

## 🔧 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/himanalot/energyrag.git
   cd energyrag
