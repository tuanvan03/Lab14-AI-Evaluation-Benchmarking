from __future__ import annotations
from typing import Any, Callable
from .model import Document
import chromadb 
import os

class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        db_path: str = "./chroma_db"
    ) -> None:
        if embedding_fn is None:
            raise ValueError("embedding_fn must be provided")
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        
        try:
            if not os.path.exists(db_path):
                os.makedirs(db_path)
            chroma_client = chromadb.PersistentClient(path=db_path)
            self._collection = chroma_client.get_or_create_collection(name=self._collection_name)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to ChromaDB: {e}")

    def add_documents(self, docs: list[Document]) -> None:
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")
        
        ids = [doc.id for doc in docs]
        contents = [doc.content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        embeddings = [self._embedding_fn(doc.content) for doc in docs]
        self._collection.add(ids=ids, documents=contents, metadatas=metadatas, embeddings=embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")

        query_embedding = self._embedding_fn(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "embedding": results['embeddings'][0][i] if results.get('embeddings') else None,
                })
        return formatted_results

    def get_collection_size(self) -> int:
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")
        return self._collection.count()

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")

        query_embedding = self._embedding_fn(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter,
            include=["documents", "metadatas", "embeddings"]
        )
        
        formatted_results = []
        if results['ids']:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "embedding": results['embeddings'][0][i] if results.get('embeddings') else None,
                })
        return formatted_results

    def delete_document(self, doc_id: str) -> bool:
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")
        
        initial_count = self.get_collection_size()
        self._collection.delete(ids=[doc_id])
        return self.get_collection_size() < initial_count

    def get_story_chunks_by_filename(self, filename: str) -> list[dict[str, Any]]:
        """
        Lấy toàn bộ thông tin các chunk phần 'story' theo vòng lặp id định dạng:
        id=f"{filename}_story_{chunk_idx}"
        """
        if not self._collection:
            raise ConnectionError("ChromaDB collection is not available.")

        chunks = []
        chunk_idx = 0
        while True:
            doc_id = f"{filename}_story_{chunk_idx}"
            results = self._collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            # Nếu ids là list rỗng nghĩa là đã hết chunk cho file này
            if not results or not results.get('ids'):
                break
                
            chunks.append({
                "id": results['ids'][0],
                "content": results['documents'][0],
                "metadata": results['metadatas'][0]
            })
            chunk_idx += 1

        for chunk in chunks:
            del chunk['metadata']
            
        return chunks