import json
import re
import os
import asyncio
import pandas as pd
from tqdm import tqdm
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerAccuracy, ContextRelevance
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai.chat_models import ChatOpenAI

def extract_documents(retrieved_contexts: str) -> list:
    return re.findall(r'<doc>\n?(.*?)</doc>', retrieved_contexts, re.DOTALL)

async def process_batch(batch, evaluator_llm, metrics):
    evaluation_dataset = EvaluationDataset.from_list(batch)
    result =  evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        batch_size=120
    )
    return result.to_pandas()

async def main():
    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct",
        api_key=os.getenv("NOVITA_API_KEY"),
        base_url="https://api.novita.ai/v3/openai",
    )
    
    adapt_llm = LangchainLLMWrapper(ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        api_key=os.getenv("NOVITA_API_KEY"),
        base_url="https://api.novita.ai/v3/openai",
    ))

    evaluator_llm = LangchainLLMWrapper(llm)
    metrics = [AnswerAccuracy(), Faithfulness(), ContextRelevance()]

    for metric in metrics:
        adapted_prompts = await metric.adapt_prompts(language="spanish", llm=adapt_llm)
        metric.set_prompts(**adapted_prompts)
    
    os.makedirs("rag_evaluations", exist_ok=True)

    json_files = [f for f in os.listdir("rag_predictions") if f.endswith(".json")]
    
    for filename in tqdm(json_files, unit="files"):
        csv_filename = filename.replace(".json", ".csv")
        csv_path = os.path.join("rag_evaluations", csv_filename)
        
        # Check if the CSV file already exists
        if os.path.isfile(csv_path):
            print(f"CSV file {csv_filename} already exists. Skipping processing.")
            continue

        json_path = os.path.join("rag_predictions", filename)
        with open(json_path, "r", encoding="utf-8") as file:
            rag_data = json.load(file)

        dataset = rag_data["dataset"]
        for record in dataset:
            record["retrieved_contexts"] = extract_documents(record["retrieved_contexts"])

        batch_size = 40
        total_records = len(dataset)
        results = pd.DataFrame()  # Initialize an empty DataFrame to store results

        for start in tqdm(range(0, total_records, batch_size), desc=f"Processing batches from {filename}",unit="batches"):
            end = min(start + batch_size, total_records)
            batch = dataset[start:end]
            batch_results = await process_batch(batch, evaluator_llm, metrics)
            results = pd.concat([results, batch_results], ignore_index=True)
            await asyncio.sleep(50)  # Sleep to comply with API rate limits

        results.to_csv(csv_path, index=False)
        print(f"Saved evaluation results to {csv_path}")

if __name__ == "__main__":
    asyncio.run(main())



        


    







