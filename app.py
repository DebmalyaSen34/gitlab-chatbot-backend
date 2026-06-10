import streamlit as st
import os
import sys
from dotenv import load_dotenv

from cache import SemanticCache
from guardrails import is_prompt_injection, is_on_topic, verify_response_grounded
from rag import RAGController

load_dotenv()

st.set_page_config(page_title="GitLab GenAI Chatbot", page_icon="🦊", layout="wide")

if os.path.exists("style.css"):
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")
if "api_base" not in st.session_state:
    st.session_state.api_base = os.environ.get("OPENAI_API_BASE", "")
if "model_name" not in st.session_state:
    st.session_state.model_name = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
if "supabase_url" not in st.session_state:
    st.session_state.supabase_url = os.environ.get("SUPABASE_URL", "")
if "supabase_key" not in st.session_state:
    st.session_state.supabase_key = os.environ.get("SUPABASE_API_KEY", "")
if "ollama_url" not in st.session_state:
    st.session_state.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def init_rag(api_key: str, supabase_url: str, supabase_key: str, ollama_url: str = "http://localhost:11434") -> RAGController | None:
    """Initialize RAGController with error handling."""
    try:
        return RAGController(
            api_key=api_key,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            ollama_url=ollama_url,
        )
    except Exception as e:
        return None


st.sidebar.title("🦊 GitLab Chatbot Setup")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "LLM API Key", type="password", value=st.session_state.api_key
)
api_base = st.sidebar.text_input(
    "LLM API Base URL", value=st.session_state.api_base,
    placeholder="https://api.openai.com/v1"
)
model_name = st.sidebar.text_input(
    "LLM Model Name", value=st.session_state.model_name,
    placeholder="gpt-3.5-turbo"
)
supabase_url = st.sidebar.text_input(
    "Supabase URL", value=st.session_state.supabase_url
)
supabase_key = st.sidebar.text_input(
    "Supabase API Key", type="password", value=st.session_state.supabase_key
)
ollama_url = st.sidebar.text_input(
    "Ollama URL", value=st.session_state.ollama_url
)

if api_key:
    st.session_state.api_key = api_key
if api_base:
    st.session_state.api_base = api_base
if model_name:
    st.session_state.model_name = model_name
if supabase_url:
    st.session_state.supabase_url = supabase_url
if supabase_key:
    st.session_state.supabase_key = supabase_key
if ollama_url:
    st.session_state.ollama_url = ollama_url

st.sidebar.markdown("---")
st.sidebar.subheader("Connection Status")
if st.session_state.api_key:
    st.sidebar.markdown("✅ LLM API Key set")
else:
    st.sidebar.markdown("❌ LLM API Key missing")
if st.session_state.supabase_url and st.session_state.supabase_key:
    st.sidebar.markdown("✅ Supabase connected")
else:
    st.sidebar.markdown("❌ Supabase credentials missing")
if st.session_state.ollama_url:
    st.sidebar.markdown(f"✅ Ollama: `{st.session_state.ollama_url}`")
else:
    st.sidebar.markdown("❌ Ollama URL missing")

cache = SemanticCache()
rag = None
if st.session_state.api_key and st.session_state.supabase_url and st.session_state.supabase_key:
    rag = init_rag(
        st.session_state.api_key,
        st.session_state.supabase_url,
        st.session_state.supabase_key,
        st.session_state.ollama_url,
    )
    if rag is None:
        st.sidebar.error("Failed to initialize RAG controller. Check your credentials.")

tab1, tab2, tab3 = st.tabs(["💬 Chat Arena", "📊 Analytics", "⚙️ Admin"])

with tab1:
    st.title("🦊 GitLab Handbook Assistant")

    if not rag:
        st.info(
            "👋 Welcome! Please provide your **LLM API Key**, **Supabase URL**, and "
            "**Supabase API Key** in the sidebar to get started."
        )
    else:
        # Suggested query chips
        st.markdown("**💡 Try asking:**")
        chips = [
            "What are GitLab's core values?",
            "How does GitLab manage async communication?",
            "Explain the CEO shadow program",
            "What is GitLab's approach to remote work?",
            "How does GitLab handle hiring?",
        ]
        cols = st.columns(len(chips))
        for idx, chip in enumerate(chips):
            if cols[idx].button(chip, key=f"chip_{idx}"):
                st.session_state.last_click = chip

        # Chat input
        user_query = st.chat_input("Ask a question about the GitLab Handbook...")

        # Process input
        query_to_process = None
        if user_query:
            query_to_process = user_query
        elif "last_click" in st.session_state:
            query_to_process = st.session_state.pop("last_click")

        if query_to_process:
            query = query_to_process

            # 1. Input Guardrails
            if is_prompt_injection(query):
                st.error("🚫 **Access Denied:** Unsafe query pattern detected. Please rephrase your question.")
            elif not is_on_topic(query, api_key=st.session_state.api_key):
                st.warning(
                    "📌 I'm specialized in the GitLab Handbook. I can't assist with this topic. "
                    "Try asking about GitLab's values, culture, hiring, or product direction!"
                )
            else:
                # 2. Semantic Cache lookup
                with st.spinner("🔍 Searching..."):
                    try:
                        query_embedding = rag.get_query_embedding(query)
                        cached_res = cache.lookup(query_embedding)

                        if cached_res:
                            st.session_state.chat_history.append(
                                {
                                    "query": query,
                                    "response": cached_res,
                                    "cache": "HIT",
                                    "latency": 0.005,
                                    "time_to_first_token": 0.005,
                                    "chunks": [],
                                }
                            )
                        else:
                            # 3. RAG retrieval & generation
                            result = rag.query(query)
                            response_text = result["response"]

                            # 4. Output guardrail
                            is_safe = verify_response_grounded(
                                response_text,
                                result["retrieved_chunks"],
                                st.session_state.api_key,
                            )

                            if is_safe:
                                st.session_state.chat_history.append(
                                    {
                                        "query": query,
                                        "response": response_text,
                                        "cache": "MISS",
                                        "latency": result["latency"],
                                        "time_to_first_token": result["time_to_first_token"],
                                        "chunks": result["retrieved_chunks"],
                                    }
                                )
                                # Store in cache
                                cache.store(query, query_embedding, response_text)
                            else:
                                st.error(
                                    "🛡️ **Output Guardrail Triggered:** The response may contain "
                                    "unverified information. Please try rephrasing your question."
                                )
                    except Exception as e:
                        error_str = str(e)
                        if "match_data_embeddings" in error_str or "PGRST202" in error_str:
                            st.error(
                                "❌ **Database setup required.** The vector search function is missing.\n\n"
                                "Go to your **Supabase Dashboard → SQL Editor** and run the "
                                "contents of `supabase_setup.sql`. See the `supabase_setup.sql` file in the project root."
                            )
                        else:
                            st.error(f"❌ Error: {error_str}")

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["query"])
            with st.chat_message("assistant", avatar="🦊"):
                st.markdown(msg["response"])

                # Show metadata
                col1, col2, col3 = st.columns(3)
                col1.caption(f"⏱️ {msg.get('latency', 0):.2f}s")
                col2.caption(f"📦 Cache: {msg.get('cache', 'N/A')}")
                col3.caption(f"🎯 TTFT: {msg.get('time_to_first_token', 0):.2f}s")

                # Show cited chunks
                if msg.get("chunks"):
                    with st.expander("📚 Source Chunks Used"):
                        for idx, chunk in enumerate(msg["chunks"]):
                            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
                            source_url = meta.get("url", "https://handbook.gitlab.com/")
                            source_title = meta.get("title", "Handbook Page")
                            st.markdown(f"**Source {idx + 1}:** [{source_title}]({source_url})")
                            st.code(
                                chunk.get_content()[:500] + ("..." if len(chunk.get_content()) > 500 else ""),
                                language="markdown",
                            )

# ============================
# Tab 2: Analytics
# ============================
with tab2:
    st.title("📊 System Analytics Dashboard")

    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div class='metric-card'><h4>Total Latency</h4>"
                f"<h3>{last_msg.get('latency', 0):.3f}s</h3></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'><h4>Cache Status</h4>"
                f"<h3>{last_msg.get('cache', 'N/A')}</h3></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='metric-card'><h4>TTFT</h4>"
                f"<h3>{last_msg.get('time_to_first_token', 0):.3f}s</h3></div>",
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"<div class='metric-card'><h4>LLM Model</h4>"
                f"<h3>{st.session_state.model_name}</h3></div>",
                unsafe_allow_html=True,
            )

        # Chat history table
        st.subheader("Query History")
        import pandas as pd

        history_data = [
            {
                "Query": msg["query"][:60] + ("..." if len(msg["query"]) > 60 else ""),
                "Cache": msg.get("cache", "N/A"),
                "Latency (s)": f"{msg.get('latency', 0):.3f}",
                "TTFT (s)": f"{msg.get('time_to_first_token', 0):.3f}",
            }
            for msg in st.session_state.chat_history
        ]
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("Submit a query in the Chat Arena to view analytics.")

# ============================
# Tab 3: Admin
# ============================
with tab3:
    st.title("⚙️ Admin Center")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Semantic Cache")
        cache_stats = cache.get_stats()
        st.metric("Cached Entries", cache_stats["total_entries"])

        if st.button("🧹 Clear Semantic Cache"):
            cache.clear_cache()
            st.success("Cache cleared successfully!")
            st.rerun()

    with col2:
        st.subheader("🗄️ Database Info")
        st.info(
            "To update the handbook index, run:\n\n"
            "```bash\n"
            "python ingest.py \\\n"
            "  --supabase-url YOUR_URL \\\n"
            "  --supabase-key YOUR_KEY\n"
            "```\n\n"
            "Embeddings are generated locally via Ollama (`embeddinggemma`)."
        )

    st.markdown("---")
    st.subheader("🔧 Run Ingestion")
    if st.button("🔄 Trigger Ingestion Pipeline"):
        if not all([st.session_state.api_key, st.session_state.supabase_url, st.session_state.supabase_key]):
            st.error("Please provide all credentials in the sidebar first.")
        else:
            with st.spinner("Running ingestion pipeline... This may take several minutes."):
                import subprocess

                try:
                    result = subprocess.run(
                        [
                            sys.executable, "ingest.py",
                            "--supabase-url", st.session_state.supabase_url,
                            "--supabase-key", st.session_state.supabase_key,
                            "--ollama-url", st.session_state.ollama_url,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode == 0:
                        st.success("Ingestion complete!")
                        st.code(result.stdout)
                    else:
                        st.error(f"Ingestion failed:\n{result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Ingestion timed out (>10 minutes).")
                except Exception as e:
                    st.error(f"Error running ingestion: {e}")
