

# Retrieval-Augmented Generation for Causality Detection in Financial Documents

This repository explores the application of Retrieval-Augmented Generation (RAG) techniques for detecting causal relationships in Spanish financial documents, using the FinCausal 2025 dataset. The project compares various RAG configurations, evaluates their performance using RAGAS metrics, and studies the effectiveness of methods like reranking, HyDE, and query rewriting.

The work is based on a thesis investigating causality detection framed as an extractive QA task, showing that a fine-tuned generative model (SuperLeNIA) significantly outperforms traditional span extraction approaches, maintaining strong results even in complex, multi-document RAG setups.

---

## Repository Structure

- **`rag_predictions/`**  
  Contains all generated predictions for each RAG configuration, evaluated on the Spanish Subtask of FinCausal 2025 ([FinCausal 25](https://www.lllf.uam.es/wordpress/fincausal-25/)).

- **`rag_evaluations/`**  
  Stores evaluation outputs:
  - Individual RAGAS metric results for each configuration.
  - `rag_evaluation_summary.csv`: Summarizes average RAGAS metrics across all configurations.
  - `rag_sas_results.csv`: Contains Semantic Answer Similarity (SAS) and Exact Match (EM) scores for each configuration.

- **`src/`**
  - **`llm_api.py`**  
    Hosts an LLM API used to make predictions.  
    > *Note:* SuperLeNIA (a proprietary model fine-tuned on FinCausal) was used for predictions, but is not included in the repo.
  
  - **`rag.py`**  
    Implements the `RAG` class supporting different configurations:
    - Reranking
    - HyDE (Hypothetical Document Embeddings)
    - Query Rewriting
    - Uploads data to HuggingFace datasets.

  - **`generate_responses.py`**  
    Automates the generation of predictions:
    - Creates a grid of RAG configurations.
    - Calls API from `llm_api.py` for predictions.
    - Saves results as JSON files in `rag_predictions/`.

  - **`evaluate_ragas.py`**  
    Evaluates each prediction set using the [RAGAS](https://docs.ragas.io/en/latest/) framework, storing detailed metrics in `rag_evaluations/`.

  - **`format_evaluations.py`**  
    Processes individual evaluations to calculate average RAGAS scores per configuration for easier comparison.

  - **`ragas_sas_em.py`**  
    Computes the SAS and Exact Match scores from the RAGAS outputs and summarizes them.  
    > *Note:* This script contains the exact implementation of the **Find closest answer algorithm** described in the thesis.

---

## Summary of Results

| Retrieval            | Rerank | Top K | Accuracy | Faithfulness | Context Relevance | SAS  | EM   | Avg. Score |
|:---------------------|:-------|:------|:---------|:-------------|:------------------|:-----|:-----|:-----------|
| Original Query        | True   | 10    | 0.86     | 0.83         | 0.99              | 0.92 | 0.73 | 0.87       |
| Original Query        | True   | 5     | 0.84     | 0.84         | 0.98              | 0.91 | 0.71 | 0.86       |
| Original Query        | False  | 10    | 0.85     | 0.82         | 0.99              | 0.92 | 0.72 | 0.86       |
| Original Query        | False  | 5     | 0.83     | 0.81         | 0.98              | 0.90 | 0.69 | 0.84       |
| HyDE                  | True   | 10    | 0.80     | 0.80         | 0.98              | 0.87 | 0.67 | 0.82       |
| HyDE                  | False  | 10    | 0.79     | 0.80         | 0.98              | 0.87 | 0.66 | 0.82       |
| HyDE                  | True   | 5     | 0.77     | 0.81         | 0.96              | 0.86 | 0.65 | 0.81       |
| HyDE                  | False  | 5     | 0.77     | 0.79         | 0.97              | 0.85 | 0.65 | 0.81       |
| Query Rewriting       | False  | 10    | 0.56     | 0.73         | 0.76              | 0.68 | 0.46 | 0.64       |
| Query Rewriting       | True   | 10    | 0.57     | 0.75         | 0.72              | 0.69 | 0.47 | 0.64       |
| Query Rewriting       | True   | 5     | 0.55     | 0.70         | 0.71              | 0.67 | 0.46 | 0.62       |
| Query Rewriting       | False  | 5     | 0.55     | 0.69         | 0.66              | 0.66 | 0.45 | 0.60       |

---

## Key Findings

The optimal RAG configuration involves using the original query to retrieve the top 10 most relevant documents, followed by reranking to select the top 3 documents for answer generation. This setup achieved the highest overall performance, with an average score of **0.87**, accuracy of **0.86**, near-perfect context relevance (**0.99**), and top SAS (**0.92**) and Exact Match (**0.73**) scores.  
Despite working with a much larger corpus (400 documents in the vector database), performance only slightly declined compared to the controlled single-context setup, demonstrating that causality detection remains highly effective even in realistic multi-document retrieval scenarios.

