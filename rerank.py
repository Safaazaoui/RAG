import os
from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class CohereReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes a lightweight cross-encoder model for reranking retrieved chunks.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document], top_k: int = 4) -> List[Document]:
        """
        Reranks a list of candidate documents relative to the user query.
        """
        if not documents:
            return []

        # Pair query with each retrieved document content
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        # Attach scores to documents and sort in descending order
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        return [doc for doc, score in scored_docs[:top_k]]