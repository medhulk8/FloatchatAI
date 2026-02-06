"""
batch_embedder.py
Batch embedding generation for better performance
"""
from typing import List, Dict, Any, Optional
import structlog
import numpy as np
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class BatchEmbedder:
    """
    Efficient batch embedding generation

    Benefits:
    - 3-5x faster than sequential embedding
    - Better GPU utilization
    - Reduced API calls if using external embedding service
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize batch embedder

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._model = None
        self.embedding_dim = 384  # For all-MiniLM-L6-v2
        self.batch_size = 32  # Optimal batch size for most GPUs

    @property
    def model(self):
        """Lazy load model to avoid initialization overhead"""
        if self._model is None:
            logger.info("Loading embedding model", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        return self._model

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts

        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 32)
            show_progress: Show progress bar

        Returns:
            List of embedding vectors (384-D each)
        """

        if not texts:
            return []

        batch_size = batch_size or self.batch_size

        logger.info(
            "Generating batch embeddings",
            text_count=len(texts),
            batch_size=batch_size
        )

        try:
            # Use sentence-transformers batch encoding
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = embeddings.tolist()

            logger.info(
                "Batch embeddings generated",
                count=len(embeddings_list),
                dimension=len(embeddings_list[0]) if embeddings_list else 0
            )

            return embeddings_list

        except Exception as e:
            logger.error("Batch embedding generation failed", error=str(e))
            raise

    def embed_documents_for_chromadb(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = 'text'
    ) -> Dict[str, Any]:
        """
        Embed documents for ChromaDB insertion

        Args:
            documents: List of document dicts
            text_field: Field name containing text to embed

        Returns:
            {
                "embeddings": List[List[float]],
                "texts": List[str],
                "metadatas": List[Dict],
                "ids": List[str]
            }
        """

        if not documents:
            return {
                "embeddings": [],
                "texts": [],
                "metadatas": [],
                "ids": []
            }

        # Extract texts
        texts = [doc.get(text_field, '') for doc in documents]

        # Generate embeddings in batch
        embeddings = self.embed_batch(texts)

        # Extract metadata and IDs
        metadatas = [doc.get('metadata', {}) for doc in documents]
        ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(documents)]

        logger.info(
            "ChromaDB batch prepared",
            document_count=len(documents),
            embedding_count=len(embeddings)
        )

        return {
            "embeddings": embeddings,
            "texts": texts,
            "metadatas": metadatas,
            "ids": ids
        }

    def embed_profiles_batch(
        self,
        profiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Embed ARGO profiles in batch with summaries

        Args:
            profiles: List of ARGO profile dicts

        Returns:
            List of profiles with added 'embedding' field
        """

        # Generate profile summaries
        summaries = [self._create_profile_summary(p) for p in profiles]

        # Batch embed all summaries
        embeddings = self.embed_batch(summaries)

        # Add embeddings to profiles
        for profile, embedding in zip(profiles, embeddings):
            profile['embedding'] = embedding
            profile['summary_text'] = self._create_profile_summary(profile)

        logger.info(
            "Profiles batch embedded",
            profile_count=len(profiles),
            avg_embedding_dim=np.mean([len(e) for e in embeddings])
        )

        return profiles

    def _create_profile_summary(self, profile: Dict[str, Any]) -> str:
        """Create text summary of ARGO profile for embedding"""

        summary_parts = []

        # Location
        if 'latitude' in profile and 'longitude' in profile:
            summary_parts.append(
                f"Location: {profile['latitude']:.2f}°, {profile['longitude']:.2f}°"
            )

        # Region
        if 'region' in profile:
            summary_parts.append(f"Region: {profile['region']}")

        # Date
        if 'profile_date' in profile or 'date' in profile:
            date = profile.get('profile_date') or profile.get('date')
            summary_parts.append(f"Date: {date}")

        # Temperature
        if 'temperature' in profile:
            temp = profile['temperature']
            if isinstance(temp, list) and temp:
                summary_parts.append(
                    f"Surface temperature: {temp[0]:.1f}°C"
                )
                if len(temp) > 1:
                    summary_parts.append(
                        f"Temperature range: {min(temp):.1f}°C to {max(temp):.1f}°C"
                    )

        # Salinity
        if 'salinity' in profile:
            sal = profile['salinity']
            if isinstance(sal, list) and sal:
                summary_parts.append(
                    f"Surface salinity: {sal[0]:.1f} PSU"
                )

        # Depth
        if 'depth' in profile or 'pressure' in profile:
            depth = profile.get('depth') or profile.get('pressure')
            if isinstance(depth, list) and depth:
                summary_parts.append(f"Maximum depth: {max(depth):.0f}m")

        # BGC data
        if profile.get('has_bgc'):
            bgc_params = []
            if 'oxygen' in profile:
                bgc_params.append('oxygen')
            if 'ph' in profile:
                bgc_params.append('pH')
            if 'nitrate' in profile:
                bgc_params.append('nitrate')
            if 'chlorophyll' in profile:
                bgc_params.append('chlorophyll')

            if bgc_params:
                summary_parts.append(
                    f"BGC parameters: {', '.join(bgc_params)}"
                )

        return " | ".join(summary_parts)

    def compute_similarity(
        self,
        query_embedding: List[float],
        document_embeddings: List[List[float]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Compute cosine similarity between query and documents

        Args:
            query_embedding: Query vector
            document_embeddings: List of document vectors
            top_k: Number of top results to return

        Returns:
            List of {index, similarity} dicts sorted by similarity
        """

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(document_embeddings)

        # Compute cosine similarity
        # similarity = (A · B) / (||A|| × ||B||)
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)

        similarities = np.dot(doc_vecs, query_vec) / (doc_norms * query_norm)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            {
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "distance": float(1 - similarities[idx])  # Convert to distance
            }
            for idx in top_indices
        ]

        logger.info(
            "Similarity computation completed",
            top_k=top_k,
            max_similarity=results[0]['similarity'] if results else 0
        )

        return results

    def warmup(self):
        """Warm up the model with a dummy encoding"""
        logger.info("Warming up embedding model")
        _ = self.model.encode(["warmup text"], show_progress_bar=False)
        logger.info("Model warmup completed")


# Global instance
batch_embedder = BatchEmbedder()
