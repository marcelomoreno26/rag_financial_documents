import itertools
import json
import os
from tqdm import tqdm
from rag import RAG
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser



if __name__=="__main__":
    top_k = [5, 10]
    retrieval_type = ["none", "HyDE", "query_rewriting"]
    rerank = [True, False]

    combinations = list(itertools.product(top_k, retrieval_type, rerank))
    grid = [{"top_k": k, "retrieval_technique": r, "rerank": re} for k, r, re in combinations]


    evaluation_dataset = concatenate_datasets([load_dataset("LenguajeNaturalAI/fin-causal-qa", split="test"), load_dataset("LenguajeNaturalAI/fin-causal-qa", split="validation")])
    queries = evaluation_dataset["question"]
    references = [answer["text"][0] for answer in evaluation_dataset["answers"]]
    rag = RAG()
    rag.upload_hf("LenguajeNaturalAI/fin-causal-qa", split="test", column="context")
    rag.upload_hf("LenguajeNaturalAI/fin-causal-qa", split="validation", column="context")

    for config in tqdm(grid):
        rag.set_config(**config)
        documents = rag.retrieve_documents(queries=queries)
        responses = rag.generate_answers(queries=queries, documents=documents, sampling_params={"temperature":0.56, "top_p":0.74, "min_p":0.01, "max_tokens":512})

        dataset = [{
                    "user_input":query,
                    "retrieved_contexts":relevant_docs,
                    "response":response,
                    "reference":reference
                } 
                for query, relevant_docs, response, reference in zip(queries, documents, responses, references)
                ]
        

        filename = f"{config['retrieval_technique']}_rerank{config['rerank']}_top_k{config['top_k']}.json"

        output_data = {
            "config": config,
            "dataset": dataset
        }

        folder_path = "rag_predictions"
        os.makedirs(folder_path, exist_ok=True) 

        filename = os.path.join(folder_path, filename)

        with open(filename, "w") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {filename}")


    