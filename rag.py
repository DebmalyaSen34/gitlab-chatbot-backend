import os
import time
import json
from openai import OpenAI as OpenAIClient
from supabase import create_client, Client
from llama_index.core.schema import NodeWithScore, TextNode


# Shared vecs resources — lazy-initialized
_vecs_collection = None
_embed_model = None


def _get_embed_model():
    """Get or create a shared fastembed model."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embed_model


def get_vecs_collection():
    """Get or create a shared vecs collection."""
    global _vecs_collection
    if _vecs_collection is None:
        import vecs
        db_connection = os.environ.get("SUPABASE_DB_CONNECTION", "")
        if not db_connection:
            raise ValueError(
                "SUPABASE_DB_CONNECTION env var required for vecs. "
                "Format: postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres"
            )
        vx = vecs.create_client(db_connection)
        _vecs_collection = vx.get_or_create_collection(
            name="handbook_embeddings",
            dimension=384,
        )
    return _vecs_collection


class RAGController:
    """Retrieval-Augmented Generation controller using vecs + OpenAI-compatible LLM."""

    def __init__(self, api_key: str, postgres_connection_string: str = "", supabase_url: str = "", supabase_key: str = "", **kwargs):
        self.api_key = api_key

        # Configure OpenAI-compatible client
        self.llm_model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        client_kwargs = {"api_key": api_key}
        llm_base_url = os.environ.get("OPENAI_API_BASE")
        if llm_base_url:
            client_kwargs["base_url"] = llm_base_url
        self.llm_client = OpenAIClient(**client_kwargs)

        # Keep Supabase client for any non-vector operations
        if supabase_url and supabase_key:
            self.supabase: Client = create_client(
                supabase_url=supabase_url, supabase_key=supabase_key
            )
        else:
            raise ValueError("supabase_url and supabase_key required")

    def get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for the query using fastembed (local model)."""
        model = _get_embed_model()
        embeddings = list(model.embed([query]))
        return embeddings[0].tolist()

    def vector_search(self, query_embedding: list[float], top_k: int = 15) -> list[dict]:
        """Search vecs collection for similar embeddings."""
        collection = get_vecs_collection()
        results = collection.query(
            data=query_embedding,
            limit=top_k,
            include_value=True,
            include_metadata=True,
        )
        # vecs with include_value=True, include_metadata=True returns (id, metadata, distance)
        search_results = []
        for result in results:
            if len(result) == 3:
                doc_id, metadata, distance = result
            elif len(result) == 2:
                doc_id, distance = result
                metadata = {}
            else:
                continue
            similarity = 1 - float(distance)
            content = metadata.get("content", "") if metadata else ""
            search_results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "file_path": metadata.get("file_path", "") if metadata else "",
                "similarity": similarity,
            })

        return search_results

    def build_nodes_from_results(self, search_results: list[dict]) -> list[NodeWithScore]:
        """Convert search results to LlamaIndex NodeWithScore objects."""
        nodes = []
        for result in search_results:
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
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

    def select_top_nodes(
        self, nodes: list[NodeWithScore], top_n: int = 5
    ) -> list[NodeWithScore]:
        """Select top-N nodes by vector similarity score."""
        if not nodes:
            return []
        sorted_nodes = sorted(nodes, key=lambda n: n.score or 0, reverse=True)
        return sorted_nodes[:top_n]

    def query(self, query_str: str, query_embedding: list[float] = None) -> dict:
        """Full RAG pipeline: embed → search → select top → generate."""
        start_time = time.time()

        # 1. Generate query embedding
        if query_embedding is None:
            query_embedding = self.get_query_embedding(query_str)

        # 2. Vector search via vecs
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

        # 3. Select top nodes by similarity score
        reranked_nodes = self.select_top_nodes(retrieved_nodes, top_n=8)

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
            "You are the GitLab Company Handbook Chatbot. You are an expert on GitLab's "
            "handbook, policies, culture, and practices. You must answer the user's question "
            "comprehensively based ONLY on the provided context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Query: {query_str}\n\n"
            "Instructions:\n"
            "1. Give a **detailed, thorough answer**. Do NOT be vague or overly brief. "
            "Extract and present all relevant information from the context — specific policies, "
            "steps, names, values, processes, and examples.\n"
            "2. Structure your answer with clear paragraphs or bullet points for readability.\n"
            "3. Cite your sources inline using markdown links (e.g. [Page Title](url)) "
            "extracted from the source metadata.\n"
            "4. If the context only partially answers the question, share everything you "
            "found and clearly state what aspects are not covered.\n"
            "5. Only say 'I cannot find this information in the GitLab handbook' if the "
            "context truly contains NOTHING relevant to the question.\n"
            "6. Do NOT tell the user to 'visit the page for more details' — you ARE the "
            "handbook assistant, so provide the details yourself.\n\n"
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
