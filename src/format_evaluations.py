import os
import re
import pandas as pd
import numpy as np

folder_path = "rag_evaluations"

pattern = re.compile(r"(?P<retrieval_technique>none|HyDE|query_rewriting)_rerank(?P<rerank>True|False)_top_k(?P<top_k>\d+)\.csv")

evaluation = []

for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        retrieval_technique = match.group("retrieval_technique")
        rerank = match.group("rerank") == "True"
        top_k = int(match.group("top_k"))

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        nv_accuracy = round(df["nv_accuracy"].mean(skipna=True), 2)
        faithfulness = round(df["faithfulness"].mean(skipna=True), 2)
        nv_context_relevance = round(df["nv_context_relevance"].mean(skipna=True), 2)
        average_score = np.mean([nv_accuracy, faithfulness, nv_context_relevance])

        evaluation.append({
            "retrieval_technique": retrieval_technique,
            "rerank": rerank,
            "top_k": top_k,
            "nv_accuracy": nv_accuracy,
            "faithfulness": faithfulness,
            "nv_context_relevance": nv_context_relevance,
            "average_score": average_score
        })


evaluation_df = pd.DataFrame(evaluation).sort_values(by="average_score", ascending=False)
evaluation_df.to_csv(os.path.join(folder_path, "rag_evaluation_summary.csv"), index=False)

        





