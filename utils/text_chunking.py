"""
Utility functions for text chunking and retrieval using TF-IDF and BM25.
"""
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# These will be imported conditionally to avoid errors if not installed
# from sklearn.feature_extraction.text import TfidfVectorizer
# from rank_bm25 import BM25Okapi
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords


class TextChunker:
    """Class for splitting documents into meaningful chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        split_by: str = "paragraph",
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            split_by: Method to split text ('paragraph', 'sentence', 'fixed')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_by = split_by

        # Check if nltk is available for sentence tokenization
        self.nltk_available = False
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Download necessary NLTK data if not already downloaded
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            self.nltk_available = True
        except ImportError:
            print("NLTK not available. Using regex-based sentence splitting.")

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text: Text to split

        Returns:
            List of paragraphs
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if self.nltk_available:
            try:
                from nltk.tokenize import sent_tokenize

                return sent_tokenize(text)
            except Exception as e:
                print(f"Error using NLTK sentence tokenization: {e}")

        # Fallback to regex-based sentence splitting
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_fixed_size(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Text to split

        Returns:
            List of fixed-size chunks
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            # Calculate end position
            end = min(start + self.chunk_size, text_len)

            # If not at the end of text and not at a whitespace, move back to the last whitespace
            if end < text_len and not text[end].isspace():
                # Find the last whitespace within the chunk
                last_space = text.rfind(" ", start, end)
                if last_space != -1:
                    end = last_space + 1

            # Add the chunk
            chunks.append(text[start:end].strip())

            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap if end < text_len else text_len

        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks based on the configured method.

        Args:
            text: Text to split into chunks

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Initial splitting based on method
        if self.split_by == "paragraph":
            segments = self._split_into_paragraphs(text)
        elif self.split_by == "sentence":
            segments = self._split_into_sentences(text)
        else:  # fixed
            return [
                {"text": chunk, "index": i, "method": "fixed"}
                for i, chunk in enumerate(self._split_fixed_size(text))
            ]

        # Combine segments into chunks that respect the chunk_size
        chunks = []
        current_chunk = ""
        current_segments = []

        for segment in segments:
            # If adding this segment would exceed chunk_size, finalize the current chunk
            if len(current_chunk) + len(segment) > self.chunk_size and current_chunk:
                chunks.append(
                    {
                        "text": current_chunk,
                        "index": len(chunks),
                        "method": self.split_by,
                        "segments": current_segments,
                    }
                )
                # Start a new chunk with overlap
                # Find segments that fit within the overlap size
                overlap_text = ""
                overlap_segments = []
                for seg in reversed(current_segments):
                    if len(overlap_text) + len(seg) <= self.chunk_overlap:
                        overlap_text = seg + " " + overlap_text
                        overlap_segments.insert(0, seg)
                    else:
                        break

                current_chunk = overlap_text.strip()
                current_segments = overlap_segments.copy()

            # Add the segment to the current chunk
            if current_chunk:
                current_chunk += " " + segment
            else:
                current_chunk = segment
            current_segments.append(segment)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(
                {
                    "text": current_chunk,
                    "index": len(chunks),
                    "method": self.split_by,
                    "segments": current_segments,
                }
            )

        return chunks

    def chunk_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document and split its text into chunks.

        Args:
            document: Document dictionary with text and metadata

        Returns:
            Document with added chunks
        """
        # Create a copy of the document to avoid modifying the original
        doc_copy = document.copy()

        # Chunk the text
        chunks = self.chunk_text(document["text"])

        # Add chunks to the document
        doc_copy["chunks"] = chunks
        doc_copy["chunk_count"] = len(chunks)

        return doc_copy


class TFIDFRetriever:
    """Class for retrieving document chunks using TF-IDF scoring."""

    def __init__(self, use_sklearn: bool = True):
        """
        Initialize the TF-IDF retriever.

        Args:
            use_sklearn: Whether to use scikit-learn's TfidfVectorizer (if available)
        """
        self.documents = []
        self.chunks = []
        self.use_sklearn = use_sklearn
        self.sklearn_available = False
        self.vectorizer = None
        self.tfidf_matrix = None

        # Check if sklearn is available
        if use_sklearn:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                self.sklearn_available = True
            except ImportError:
                print("scikit-learn not available. Using custom TF-IDF implementation.")
                self.use_sklearn = False

        # Initialize custom TF-IDF if not using sklearn
        if not self.use_sklearn:
            self.document_freq = Counter()
            self.idf = {}
            self.doc_vectors = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization by splitting on whitespace and removing punctuation
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _compute_custom_tfidf(self):
        """Compute TF-IDF scores using custom implementation."""
        # Calculate document frequency
        self.document_freq = Counter()
        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk["text"]))
            for token in tokens:
                self.document_freq[token] += 1

        # Calculate IDF
        num_docs = len(self.chunks)
        self.idf = {
            token: math.log(num_docs / (freq + 1)) + 1
            for token, freq in self.document_freq.items()
        }

        # Calculate TF-IDF vectors for each document
        self.doc_vectors = []
        for chunk in self.chunks:
            tokens = self._tokenize(chunk["text"])
            term_freq = Counter(tokens)
            total_terms = len(tokens)

            # Normalize term frequency by document length
            term_freq = {term: freq / total_terms for term, freq in term_freq.items()}

            # Calculate TF-IDF
            tfidf_vector = {
                term: freq * self.idf.get(term, 0) for term, freq in term_freq.items()
            }
            self.doc_vectors.append(tfidf_vector)

    def _compute_sklearn_tfidf(self):
        """Compute TF-IDF scores using scikit-learn."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Extract text from chunks
        texts = [chunk["text"] for chunk in self.chunks]

        # Create and fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def add_document(self, document: Dict[str, Any]):
        """
        Add a document with chunks to the retriever.

        Args:
            document: Document dictionary with chunks
        """
        if "chunks" not in document:
            raise ValueError("Document must contain chunks. Use TextChunker first.")

        self.documents.append(document)
        doc_id = document.get("id", str(len(self.documents)))

        # Add document ID to each chunk
        for chunk in document["chunks"]:
            chunk_with_id = chunk.copy()
            chunk_with_id["doc_id"] = doc_id
            self.chunks.append(chunk_with_id)

        # Recompute TF-IDF
        if self.use_sklearn and self.sklearn_available:
            self._compute_sklearn_tfidf()
        else:
            self._compute_custom_tfidf()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks relevant to the query using TF-IDF.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks:
            return []

        if self.use_sklearn and self.sklearn_available:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query])

            # Calculate cosine similarity between query and documents
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top-k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            results = []

            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(similarities[idx])
                results.append(chunk)

            return results
        else:
            # Tokenize query
            query_tokens = self._tokenize(query)
            query_tf = Counter(query_tokens)
            total_query_terms = len(query_tokens)

            # Normalize term frequency
            query_tf = {
                term: freq / total_query_terms for term, freq in query_tf.items()
            }

            # Calculate query TF-IDF
            query_tfidf = {
                term: freq * self.idf.get(term, 0) for term, freq in query_tf.items()
            }

            # Calculate cosine similarity with each document
            similarities = []
            for idx, doc_vector in enumerate(self.doc_vectors):
                # Calculate dot product
                dot_product = sum(
                    query_tfidf.get(term, 0) * score
                    for term, score in doc_vector.items()
                )

                # Calculate magnitudes
                query_magnitude = math.sqrt(
                    sum(score ** 2 for score in query_tfidf.values())
                )
                doc_magnitude = math.sqrt(
                    sum(score ** 2 for score in doc_vector.values())
                )

                # Calculate cosine similarity
                similarity = (
                    dot_product / (query_magnitude * doc_magnitude)
                    if query_magnitude > 0 and doc_magnitude > 0
                    else 0
                )
                similarities.append((idx, similarity))

            # Sort by similarity and get top-k
            top_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            results = []

            for idx, score in top_results:
                chunk = self.chunks[idx].copy()
                chunk["score"] = score
                results.append(chunk)

            return results


class BM25Retriever:
    """Class for retrieving document chunks using BM25 scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_library: bool = True):
        """
        Initialize the BM25 retriever.

        Args:
            k1: BM25 parameter for term frequency scaling
            b: BM25 parameter for document length normalization
            use_library: Whether to use rank_bm25 library (if available)
        """
        self.documents = []
        self.chunks = []
        self.k1 = k1
        self.b = b
        self.use_library = use_library
        self.library_available = False
        self.bm25 = None
        self.tokenized_corpus = []

        # Check if rank_bm25 is available
        if use_library:
            try:
                from rank_bm25 import BM25Okapi

                self.library_available = True
            except ImportError:
                print("rank_bm25 not available. Using custom BM25 implementation.")
                self.use_library = False

        # Initialize custom BM25 if not using library
        if not self.use_library:
            self.doc_freqs = []
            self.doc_lens = []
            self.avg_doc_len = 0
            self.corpus_size = 0
            self.idf = {}

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Simple tokenization by splitting on whitespace and removing punctuation
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _compute_custom_bm25(self):
        """Compute BM25 parameters using custom implementation."""
        # Tokenize all documents
        self.tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]

        # Calculate document lengths
        self.doc_lens = [len(tokens) for tokens in self.tokenized_corpus]
        self.corpus_size = len(self.tokenized_corpus)
        self.avg_doc_len = (
            sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        )

        # Calculate document frequencies
        self.doc_freqs = defaultdict(int)
        for tokens in self.tokenized_corpus:
            for token in set(tokens):
                self.doc_freqs[token] += 1

        # Calculate IDF
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1
            )

    def _compute_library_bm25(self):
        """Compute BM25 using rank_bm25 library."""
        from rank_bm25 import BM25Okapi

        # Tokenize all documents
        self.tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]

        # Create BM25 object
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

    def add_document(self, document: Dict[str, Any]):
        """
        Add a document with chunks to the retriever.

        Args:
            document: Document dictionary with chunks
        """
        if "chunks" not in document:
            raise ValueError("Document must contain chunks. Use TextChunker first.")

        self.documents.append(document)
        doc_id = document.get("id", str(len(self.documents)))

        # Add document ID to each chunk
        for chunk in document["chunks"]:
            chunk_with_id = chunk.copy()
            chunk_with_id["doc_id"] = doc_id
            self.chunks.append(chunk_with_id)

        # Recompute BM25
        if self.use_library and self.library_available:
            self._compute_library_bm25()
        else:
            self._compute_custom_bm25()

    def _score_custom_bm25(self, query_tokens: List[str]) -> List[float]:
        """
        Score documents using custom BM25 implementation.

        Args:
            query_tokens: Tokenized query

        Returns:
            List of BM25 scores for each document
        """
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # Get IDF for this term
            idf = self.idf[token]

            for doc_idx, doc_tokens in enumerate(self.tokenized_corpus):
                # Count term frequency in document
                term_freq = doc_tokens.count(token)
                if term_freq == 0:
                    continue

                # Calculate BM25 score for this term in this document
                doc_len = self.doc_lens[doc_idx]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_len
                )
                scores[doc_idx] += idf * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks relevant to the query using BM25.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        if self.use_library and self.library_available:
            # Get scores using rank_bm25 library
            scores = self.bm25.get_scores(query_tokens)
        else:
            # Get scores using custom implementation
            scores = self._score_custom_bm25(query_tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []

        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results


class HybridRetriever:
    """Class for retrieving document chunks using a combination of TF-IDF and BM25."""

    def __init__(
        self,
        tfidf_weight: float = 0.5,
        use_sklearn: bool = True,
        use_library: bool = True,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            tfidf_weight: Weight for TF-IDF scores (1-tfidf_weight for BM25)
            use_sklearn: Whether to use scikit-learn for TF-IDF
            use_library: Whether to use rank_bm25 library for BM25
        """
        self.tfidf_retriever = TFIDFRetriever(use_sklearn=use_sklearn)
        self.bm25_retriever = BM25Retriever(use_library=use_library)
        self.tfidf_weight = tfidf_weight
        self.documents = []

    def add_document(self, document: Dict[str, Any]):
        """
        Add a document with chunks to both retrievers.

        Args:
            document: Document dictionary with chunks
        """
        if "chunks" not in document:
            raise ValueError("Document must contain chunks. Use TextChunker first.")

        self.documents.append(document)
        self.tfidf_retriever.add_document(document)
        self.bm25_retriever.add_document(document)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for chunks relevant to the query using both TF-IDF and BM25.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with combined scores
        """
        # Get results from both retrievers
        tfidf_results = self.tfidf_retriever.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        # Combine results
        combined_results = {}

        # Process TF-IDF results
        for result in tfidf_results:
            doc_id = result["doc_id"]
            chunk_idx = result["index"]
            key = f"{doc_id}_{chunk_idx}"

            if key not in combined_results:
                combined_results[key] = result.copy()
                combined_results[key]["tfidf_score"] = result["score"]
                combined_results[key]["bm25_score"] = 0.0
                combined_results[key]["combined_score"] = (
                    self.tfidf_weight * result["score"]
                )
            else:
                combined_results[key]["tfidf_score"] = result["score"]
                combined_results[key]["combined_score"] += (
                    self.tfidf_weight * result["score"]
                )

        # Process BM25 results
        for result in bm25_results:
            doc_id = result["doc_id"]
            chunk_idx = result["index"]
            key = f"{doc_id}_{chunk_idx}"

            if key not in combined_results:
                combined_results[key] = result.copy()
                combined_results[key]["bm25_score"] = result["score"]
                combined_results[key]["tfidf_score"] = 0.0
                combined_results[key]["combined_score"] = (
                    1 - self.tfidf_weight
                ) * result["score"]
            else:
                combined_results[key]["bm25_score"] = result["score"]
                combined_results[key]["combined_score"] += (
                    1 - self.tfidf_weight
                ) * result["score"]

        # Sort by combined score and get top-k
        results = list(combined_results.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results[:top_k]


def get_contextual_chunks(
    query: str, documents: List[Dict[str, Any]], top_k: int = 5, method: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Get contextual chunks from documents based on a query.

    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of top chunks to return
        method: Retrieval method ('tfidf', 'bm25', or 'hybrid')

    Returns:
        List of relevant chunks with scores
    """
    # Create a text chunker
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200, split_by="paragraph")

    # Create the appropriate retriever
    if method == "tfidf":
        retriever = TFIDFRetriever()
    elif method == "bm25":
        retriever = BM25Retriever()
    else:  # hybrid
        retriever = HybridRetriever()

    # Process and add each document
    for document in documents:
        # Skip if document has no text
        if "text" not in document or not document["text"]:
            continue

        # Chunk the document if not already chunked
        if "chunks" not in document:
            document = chunker.chunk_document(document)

        # Add to retriever
        retriever.add_document(document)

    # Search for relevant chunks
    results = retriever.search(query, top_k=top_k)

    return results
