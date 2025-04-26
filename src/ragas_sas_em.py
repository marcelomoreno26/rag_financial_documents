import os
import re
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
from polyfuzz import PolyFuzz
from nltk import ngrams
from polyfuzz.models import RapidFuzz




def generate_ngrams(text, n):
    words = text.split()
    return [' '.join(gram) for gram in ngrams(words, n)]



def find_closest_answer(context, prediction, delta=2):
    # Clean the context
    try:
        clean_context = context.replace('<doc>', ' ').replace('</doc>', ' ').replace('\n', ' ')
        clean_context = ' '.join(clean_context.split())  # Remove double/multiple spaces

        matches = []
        for n in range(max(1, len(prediction.split()) - delta), len(prediction.split()) + delta + 1):
            context_ngrams = generate_ngrams(clean_context, n)
            model = PolyFuzz(RapidFuzz())
            model.match([prediction], context_ngrams)
            match = model.get_matches()
            matches.append((match['To'][0], match['Similarity'][0]))

        result = max(matches, key=lambda x: x[1])[0]

        return result.rstrip(".,") if prediction and prediction[-1] != "." else result
    except:
        return prediction

    


def evaluate(st_model, predictions, references):
    """
    Evalúa las predicciones en comparación con las referencias utilizando similaridad de coseno y exact match.

    Args:
        predictions (list): Lista de respuestas generadas por el modelo.
        references (list): Lista de respuestas de referencia.

    Returns:
        dict: Diccionario con los puntajes de similaridad promedio (SAS) y exact match.
    """
    predictions_embeddings = st_model.encode(predictions, convert_to_tensor=True)
    reference_embeddings = st_model.encode(references, convert_to_tensor=True)
    similarity_scores = [util.cos_sim(p, l).cpu().numpy() for p, l in zip(predictions_embeddings, reference_embeddings)]

    sas = np.mean(similarity_scores)
    exact_match = np.mean([1 if pred.lower().strip() == ref.lower().strip() else 0 for pred, ref in zip(predictions, references)])

    return {"sas": float(sas), "exact_match": float(exact_match)}


if __name__ == '__main__':
    st_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    folder_path = "rag_predictions"

    pattern = re.compile(r"(?P<retrieval_technique>none|HyDE|query_rewriting)_rerank(?P<rerank>True|False)_top_k(?P<top_k>\d+)\.json")

    results = []

    files = [f for f in os.listdir(folder_path) if pattern.match(f)]

    for filename in tqdm(files, desc="Evaluating files", unit="file"):
        match = pattern.match(filename)
        if match:
            retrieval_technique = match.group("retrieval_technique")
            rerank = match.group("rerank") == "True"
            top_k = int(match.group("top_k"))

            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            predictions = []
            references = []

            for item in data['dataset']:
                context = item['retrieved_contexts']
                prediction = item['response']
                reference = item['reference']

                cleaned_prediction = find_closest_answer(context, prediction)
                predictions.append(prediction)
                references.append(reference)

            scores = evaluate(st_model, predictions, references)

            result = {
                "retrieval_technique": retrieval_technique,
                "rerank": rerank,
                "top_k": top_k,
                "sas": scores["sas"],
                "exact_match": scores["exact_match"]
            }
            results.append(result)

            # Print results after each file
            print(f"\nFinished evaluating {filename}:")
            print(result)
            print("-" * 50)

    # Save everything at the end
    df = pd.DataFrame(results).sort_values(by="sas", ascending=False)
    df.to_csv(os.path.join('rag_evaluations', "rag_sas_results.csv"), index=False)

