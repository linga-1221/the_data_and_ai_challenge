"""
Embedding Manager with Sentence Transformers and FAISS

Handles embedding generation, caching, and FAISS index for fast similarity search.
Optimized for 100k+ candidates with batch processing and disk persistence.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding manager"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    cache_dir: str = "./cache/embeddings"
    faiss_index_path: str = "./cache/faiss_index.bin"
    normalize_embeddings: bool = True
    use_faiss: bool = True


class EmbeddingManager:
    """
    Manages text embeddings with sentence-transformers and FAISS index.

    Features:
    - Lazy loading of model (memory efficient)
    - Disk caching of embeddings
    - FAISS index for fast similarity search
    - Batch processing for efficiency
    - Supports multiple embedding strategies
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._faiss_index = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: List[str] = []

        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Try to load cached data
        self._load_cache()

    def _get_model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.config.model_name}")
                self._model = SentenceTransformer(self.config.model_name)
                logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Install with: "
                    "pip install sentence-transformers"
                )
        return self._model

    def _compute_hash(self, text: str) -> str:
        """Compute hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for a key"""
        return os.path.join(self.config.cache_dir, f"{cache_key}.npy")

    def _load_cache(self) -> None:
        """Load embeddings and FAISS index from disk if available"""
        # Load ID mappings
        mapping_path = os.path.join(self.config.cache_dir, "id_mappings.pkl")
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'rb') as f:
                    mappings = pickle.load(f)
                    self._id_to_idx = mappings['id_to_idx']
                    self._idx_to_id = mappings['idx_to_id']
                logger.info(f"Loaded {len(self._id_to_idx)} candidate ID mappings from cache")
            except Exception as e:
                logger.warning(f"Failed to load ID mappings: {e}")

        # Load FAISS index if enabled
        if self.config.use_faiss and os.path.exists(self.config.faiss_index_path):
            try:
                import faiss
                self._faiss_index = faiss.read_index(self.config.faiss_index_path)
                logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal} vectors")
            except ImportError:
                logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
                self.config.use_faiss = False
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

    def _save_cache(self) -> None:
        """Save embeddings and FAISS index to disk"""
        # Save ID mappings
        mapping_path = os.path.join(self.config.cache_dir, "id_mappings.pkl")
        try:
            with open(mapping_path, 'wb') as f:
                pickle.dump({
                    'id_to_idx': self._id_to_idx,
                    'idx_to_id': self._idx_to_id
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save ID mappings: {e}")

        # Save FAISS index
        if self.config.use_faiss and self._faiss_index is not None:
            try:
                faiss.write_index(self._faiss_index, self.config.faiss_index_path)
                logger.info(f"Saved FAISS index with {self._faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

    def embed_texts(self, texts: List[str], normalize: bool = None) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings
            normalize: Whether to normalize embeddings (overrides config)

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        model = self._get_model()
        normalize = normalize if normalize is not None else self.config.normalize_embeddings

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=len(texts) > 100
            )
            # Cache each embedding in this batch
            for text, embedding in zip(batch, embeddings):
                cache_key = self._compute_hash(text)
                self._embeddings_cache[cache_key] = embedding
            all_embeddings.append(embeddings)

        embeddings = np.vstack(all_embeddings)
        # Cache combined batch under the hash of the concatenated texts
        combined_key = self._compute_hash(' '.join(texts))
        self._embeddings_cache[combined_key] = embeddings
        return embeddings

    def embed_and_cache(
        self,
        candidate_id: str,
        texts: List[str],
        force_refresh: bool = False
    ) -> np.ndarray:
        """
        Get embedding for a candidate, using cache if available.

        Args:
            candidate_id: Unique candidate identifier
            texts: List of text components to embed (concatenated)
            force_refresh: Force regeneration even if cached

        Returns:
            Embedding vector (1, embedding_dim)
        """
        cache_key = self._compute_hash(' '.join(texts))
        cache_path = self._get_cache_path(cache_key)

        # Check memory cache first
        if not force_refresh and cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]

        # Check disk cache
        if not force_refresh and os.path.exists(cache_path):
            try:
                embedding = np.load(cache_path)
                self._embeddings_cache[cache_key] = embedding
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")

        # Generate new embedding
        combined_text = ' '.join(texts)
        embedding = self.embed_texts([combined_text])[0]

        # Cache in memory and disk
        self._embeddings_cache[cache_key] = embedding
        try:
            np.save(cache_path, embedding)
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache: {e}")

        return embedding

    def build_faiss_index(
        self,
        candidate_ids: List[str],
        embeddings: np.ndarray,
        metric: str = 'cosine'
    ) -> None:
        """
        Build FAISS index from embeddings.

        Args:
            candidate_ids: List of candidate identifiers
            embeddings: numpy array of shape (n_candidates, embedding_dim)
            metric: Distance metric ('cosine' or 'l2')
        """
        if not self.config.use_faiss:
            logger.warning("FAISS not available. Skipping index build.")
            return

        try:
            import faiss

            n_candidates, dim = embeddings.shape

            # Normalize for cosine similarity
            if metric == 'cosine':
                faiss.normalize_L2(embeddings)
                index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
            else:
                index = faiss.IndexFlatL2(dim)  # L2 distance

            # Add vectors to index
            index.add(embeddings.astype(np.float32))

            self._faiss_index = index
            self._idx_to_id = candidate_ids
            self._id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

            logger.info(f"Built FAISS index: {n_candidates} vectors, dimension {dim}")

            # Save to disk
            self._save_cache()

        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            self.config.use_faiss = False

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar candidates using FAISS index.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return

        Returns:
            List of (candidate_id, similarity_score) tuples, sorted by score descending
        """
        if self._faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")

        # Search - note: for IndexFlatIP, higher is better (cosine similarity)
        distances, indices = self._faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, self._faiss_index.ntotal)
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._idx_to_id):
                # For IndexFlatIP, distance is inner product (cosine similarity for normalized vectors)
                similarity = float(dist)
                results.append((self._idx_to_id[idx], similarity))

        return results

    def compute_similarity_matrix(
        self,
        query_embeddings: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix between queries and candidates.

        Args:
            query_embeddings: Query embeddings (n_queries, dim)
            candidate_embeddings: Candidate embeddings (n_candidates, dim)

        Returns:
            Similarity matrix (n_queries, n_candidates)
        """
        # Normalize
        if self.config.normalize_embeddings:
            query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            candidate_norm = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            query_normalized = query_embeddings / np.maximum(query_norm, 1e-8)
            candidate_normalized = candidate_embeddings / np.maximum(candidate_norm, 1e-8)
        else:
            query_normalized = query_embeddings
            candidate_normalized = candidate_embeddings

        # Cosine similarity
        similarity = np.dot(query_normalized, candidate_normalized.T)
        return similarity

    def clear_cache(self) -> None:
        """Clear in-memory cache"""
        self._embeddings_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self._embeddings_cache),
            'indexed_candidates': len(self._idx_to_id),
            'cache_dir': self.config.cache_dir,
            'faiss_index_exists': self._faiss_index is not None,
            'faiss_index_size': self._faiss_index.ntotal if self._faiss_index else 0
        }


def create_embedding_manager(config: Optional[Dict] = None) -> EmbeddingManager:
    """
    Factory function to create EmbeddingManager with configuration.

    Args:
        config: Optional config dict with model_name, cache_dir, etc.

    Returns:
        EmbeddingManager instance
    """
    emb_config = EmbeddingConfig()
    if config:
        for key, value in config.items():
            if hasattr(emb_config, key):
                setattr(emb_config, key, value)

    return EmbeddingManager(emb_config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Test
    manager = create_embedding_manager()
    texts = ["Python developer with TensorFlow experience", "Java engineer with Spring Boot", "ML practitioner"]
    embeddings = manager.embed_texts(texts)
    print(f"Generated embeddings: shape={embeddings.shape}")

    # Build index
    candidate_ids = [f"CAND_{i:06d}" for i in range(len(texts))]
    manager.build_faiss_index(candidate_ids, embeddings)

    # Search
    query = "machine learning engineer"
    query_emb = manager.embed_texts([query])[0]
    results = manager.search_similar(query_emb, top_k=3)
    print(f"Search results: {results}")
