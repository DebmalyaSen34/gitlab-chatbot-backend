import os
import time
import requests
from openai import OpenAI as OpenAIClient
from supabase import create_client, Client
from llama_index.core.schema import NodeWithScore, TextNode


class RAGController:
    """Retrieval-Augmented Generation controller using Supabase + OpenAI-compatible LLM."""

    def __init__(self, api_key: str, postgres_connection_string: str = "", supabase_url: str = "", supabase_key: str = "", ollama_url: str = "http://localhost:11434", ollama_model: str = "embeddinggemma"):
        self.api_key = api_key
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

        # Configure OpenAI-compatible client (bypasses LlamaIndex model-name validation)
        self.llm_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        client_kwargs = {"api_key": api_key}
        llm_base_url = os.environ.get("OPENAI_API_BASE")
        if llm_base_url:
            client_kwargs["base_url"] = llm_base_url
        self.llm_client = OpenAIClient(**client_kwargs)

        # Connect to Supabase
        if supabase_url and supabase_key:
            self.supabase: Client = create_client(
                supabase_url=supabase_url, supabase_key=supabase_key
            )
        elif postgres_connection_string:
            # Fallback: create supabase from connection string components
            # This is less ideal but works for backward compatibility
            self.supabase = None
            self._postgres_connection_string = postgres_connection_string
        else:
            raise ValueError("Either supabase_url+supabase_key or postgres_connection_string required")

    def get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for the query using Ollama local model."""
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={"model": self.ollama_model, "input": [query]},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


    def vector_search(self, query_embedding: list[float], top_k: int = 15) -> list[dict]:
        """Search Supabase for similar embeddings using pgvector RPC."""
        if self.supabase:
            try:
                result = self.supabase.rpc(
                    "match_data_embeddings",
                    {
                        "query_embedding": query_embedding,
                        "match_count": top_k,
                    },
                ).execute()
                return result.data if result.data else []
            except Exception as e:
                error_msg = str(e)
                if "PGRST202" in error_msg or "match_data_embeddings" in error_msg:
                    raise RuntimeError(
                        "The match_data_embeddings function is not set up in Supabase. "
                        "Please run the SQL in supabase_setup.sql in your Supabase SQL Editor "
                        "(Dashboard → SQL Editor → New Query)."
                    ) from e
                raise
        else:
            # Direct PostgreSQL fallback
            import psycopg2
            import json

            conn = psycopg2.connect(self._postgres_connection_string)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT content, metadata, file_path,
                               1 - (embedding <=> %s::vector) as similarity
                        FROM data_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (json.dumps(query_embedding), json.dumps(query_embedding), top_k),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "content": row[0],
                            "metadata": row[1],
                            "file_path": row[2],
                            "similarity": row[3],
                        }
                        for row in rows
                    ]
            finally:
                conn.close()

    def build_nodes_from_results(self, search_results: list[dict]) -> list[NodeWithScore]:
        """Convert Supabase search results to LlamaIndex NodeWithScore objects."""
        nodes = []
        for result in search_results:
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            node = TextNode(
                text=result.get("content", ""),
                metadata=metadata,
            )
            score = result.get("similarity", 0.0)
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    def _llm_complete(self, prompt: str) -> str:
        """Call the LLM directly via the OpenAI-compatible endpoint."""
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def rerank_nodes(
        self, nodes: list[NodeWithScore], query_str: str, top_n: int = 5
    ) -> list[NodeWithScore]:
        """Rerank nodes using LLM-based scoring."""
        if not nodes:
            return []

        try:
            # Score each node with the LLM for relevance
            scored_nodes = []
            for node_with_score in nodes:
                node = node_with_score.node
                score_prompt = (
                    f"Rate the relevance of the following text to the query on a scale of 0 to 10. "
                    f"Reply with ONLY a number.\n\n"
                    f"Query: {query_str}\n\n"
                    f"Text: {node.get_content()[:1000]}"
                )
                try:
                    score_text = self._llm_complete(score_prompt)
                    score = float(score_text.strip().split()[0])
                except (ValueError, IndexError):
                    score = node_with_score.score or 0.0
                scored_nodes.append(NodeWithScore(node=node, score=score))

            scored_nodes.sort(key=lambda n: n.score or 0, reverse=True)
            reranked = scored_nodes[:top_n]
            if not reranked:
                raise ValueError("Reranker returned empty list")
            return reranked
        except Exception as e:
            print(f"Reranking failed, falling back to top-{top_n} by score: {e}")
            # Fallback: return top-n by similarity score
            sorted_nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)
            return sorted_nodes[:top_n]

    def query(self, query_str: str) -> dict:
        """Full RAG pipeline: embed → search → rerank → generate."""
        start_time = time.time()

        # 1. Generate query embedding
        query_embedding = self.get_query_embedding(query_str)

        # 2. Vector search in Supabase
        search_results = self.vector_search(query_embedding, top_k=15)

        # Filter by minimum similarity threshold
        MIN_SIMILARITY = 0.3
        search_results = [r for r in search_results if r.get("similarity", 0) >= MIN_SIMILARITY]

        if not search_results:
            return {
                "response": "I cannot find this information in the GitLab handbook.",
                "retrieved_chunks": [],
                "latency": time.time() - start_time,
                "time_to_first_token": 0,
                "num_chunks_retrieved": 0,
                "num_chunks_reranked": 0,
            }

        retrieved_nodes = self.build_nodes_from_results(search_results)

        # 3. Rerank
        reranked_nodes = self.rerank_nodes(retrieved_nodes, query_str, top_n=5)

        # 4. Build context
        context_parts = []
        for node_with_score in reranked_nodes:
            node = node_with_score.node
            meta = node.metadata if hasattr(node, "metadata") else {}
            source = meta.get("url", meta.get("source_path", "GitLab Handbook"))
            context_parts.append(f"[Source: {source}]\n{node.get_content()}")
        context_str = "\n\n---\n\n".join(context_parts)

        # 5. Generate response
        prompt = (
            "You are the GitLab Company Handbook Chatbot. You must answer the user query "
            "based ONLY on the following context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Query: {query_str}\n\n"
            "Requirements:\n"
            "1. Keep the response factual and grounded in the context.\n"
            "2. Cite your sources inline using markdown links (e.g. [Page Title](url)) "
            "extracted from the source metadata.\n"
            "3. If the context does not contain enough information to answer, say: "
            "'I cannot find this information in the GitLab handbook.'\n"
            "4. Be concise but thorough.\n\n"
            "Answer:"
        )

        t0 = time.time()
        response = self._llm_complete(prompt)
        generation_time = time.time() - t0
        total_latency = time.time() - start_time

        return {
            "response": str(response),
            "retrieved_chunks": [n.node for n in reranked_nodes],
            "latency": total_latency,
            "time_to_first_token": generation_time,
            "num_chunks_retrieved": len(retrieved_nodes),
            "num_chunks_reranked": len(reranked_nodes),
        }
