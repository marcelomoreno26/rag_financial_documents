import requests
import os
import shutil
import atexit
from datasets import load_dataset
from llama_index.core import (
    Settings,
    StorageContext, 
    load_index_from_storage,
    VectorStoreIndex
)
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank


class RAG:
    def __init__(self, embeddings_model: str= "BAAI/bge-m3", reranker_model: str="jinaai/jina-reranker-v2-base-multilingual", storage_dir: str="./index_storage", generation_port: int = 8000, retrieval_techniques_port: int=8001):
        """
        Initialize the VectorDB object.
        
        Parameters
        ----------
        embeddings_model : str
            The model name for HuggingFace embeddings.
        storage_dir : str
            Directory where the index data will be stored.
        """
        self.generation_port = generation_port
        self.retrieval_techniques_port = retrieval_techniques_port
        self.storage_dir = storage_dir
        Settings.embed_model = HuggingFaceEmbedding(model_name=embeddings_model)
        Settings.llm = None
        self.reranker = SentenceTransformerRerank(
        model=reranker_model,
        top_n=5
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        atexit.register(self._cleanup)


    def _cleanup(self):
        """
        Clean up by deleting the storage directory when the program exits.
        """
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    
    def _upload_texts(self, texts: list[str]):
        """
        Upload a list of texts into the vector database.

        Parameters
        ----------
        texts : list of str
            List of texts to be uploaded.
        """
        nodes = [TextNode(text=text) for text in texts]

        if os.listdir(self.storage_dir):
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(storage_context)
            self.index.insert_nodes(nodes)
        else:
            self.index = VectorStoreIndex(nodes)

        self.index.storage_context.persist(persist_dir=self.storage_dir)
        
    
    def _inference(self, messages: list, port: int, sampling_params: dict=None):
        """
        Sends a request to the API to generate content based on the provided messages.

        Parameters
        ----------
        messages : list of dict
            A list of message dictionaries with "role" (str) and "content" (str).
        
        port : int
            The port number where the API is hosted.

        Returns
        -------
        list or None
            A list of predictions from the API, or None if the request fails.
        """
        if sampling_params:
            response = requests.post(f"http://localhost:{port}/generate/", json={"messages": messages})
        else:
            response = requests.post(f"http://localhost:{port}/generate/", json={"messages": messages, "sampling_params":sampling_params})

        data = response.json()
        if not data.get("success"):
            print(f"API inference failed.")
            return None
        return data.get("predictions")
    

    def _get_hyde(self, queries: list[str]) -> list[str]:
        system_message = "Eres un experto en finanzas con amplia experiencia en la creación de documentos y artículos hipotéticos financieros."
        prompt = "Actúa como un experto en finanzas y redacta un párrafo de máximo 150 palabras para un artículo/documento financiero hipotético, basado en un escenario o situación ficticia. El artículo debe responder a la siguiente pregunta: {}. Devuelve solamente el artículo/documento financiero hipotético."
        
        messages = [[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": prompt.format(query)
            }
            ] 
            for query in queries]
        

        return self._inference(messages, self.retrieval_techniques_port)


    def _get_query_rewriting(self, queries: list[str]) -> list[str]:
        system_message = "Eres un experto en procesamiento de lenguaje natural y reescritura de consultas. Tu tarea es mejorar la claridad, precisión y relevancia de las consultas originales, manteniendo el sentido original."
        prompt = "Actúa como un experto en reescritura de consultas y mejora la siguiente consulta: {}\nDevuelve solamente la nueva consulta, si consideras que no se puede mejorar devuelve la consulta original."

        messages = [
            [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": prompt.format(query)
                }
            ]
            for query in queries
        ]

        return self._inference(messages, self.retrieval_techniques_port)

    
    def _rerank(self, queries: list[str], list_nodes: list[list]):
        reranked_list_nodes = []
        for query, nodes in zip(queries, list_nodes):
            query_bundle = QueryBundle(query)
            reranked_nodes = self.reranker.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
            reranked_list_nodes.append(reranked_nodes)


        return reranked_list_nodes
    

    def set_config(self, retrieval_technique: str, rerank: bool, top_k: int):
        self.retrieval_technique = retrieval_technique
        self.rerank = rerank
        self.top_k = top_k


    def upload_hf(self, dataset_name: str, split:str, column: str):
        """
        Upload a specified column of a HuggingFace dataset to the vector database.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load from HuggingFace.
        split : str
            The dataset split to load (e.g., 'train', 'validation').
        column : str
            The column in the dataset to extract for indexing.
        """
        dataset = load_dataset(dataset_name, split=split)
        texts = dataset[column]
        self._upload_texts(texts=texts)
        print(f"Column '{column}' from dataset '{dataset_name}' uploaded correctly.")


    def retrieve_documents(self, queries: list[str]) -> list[str]:
        """
        Retrieve documents for multiple queries in a batch.

        Parameters
        ----------
        queries : list of str
            List of query messages.

        Returns
        -------
        list of str
            List of concatenated strings of retrieved documents for each query.
        """
        query_engine = self.index.as_query_engine(streaming=False, similarity_top_k=self.top_k)

        if self.retrieval_technique == "HyDE":
            messages = self._get_hyde(queries=queries)
        elif self.retrieval_technique == "query_rewriting":
            messages = self._get_query_rewriting(queries=queries)
        elif self.retrieval_technique == "none":
            messages = queries
        
        
        list_nodes = [query_engine.retrieve(message) for message in messages]

        if self.rerank:
            list_nodes = self._rerank(queries=queries, list_nodes=list_nodes)

        return ["".join([f"<doc>\n{node.text}</doc>" for node in retrieved]) for retrieved in list_nodes]



    def generate_answers(self, queries: list[str], documents: list[str], sampling_params: dict=None) -> list[str]:
        messages = []

        for query, query_documents in zip(queries, documents):
            messages.append([
                {
                    "role":"system",
                    "content":"Eres un asistente que responde preguntas sobre temas financieros. Debes responder a la pregunta utilizando únicamente la información del contexto. Es decir, la respuesta debe estar extraída explícitamente del contexto de forma literal."
                },
                {
                    "role":"input",
                    "content": query_documents
                },
                {
                    "role":"user",
                    "content": query
                }
            ])
        

        return self._inference(messages=messages, port=self.generation_port, sampling_params=sampling_params)

