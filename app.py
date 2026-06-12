import streamlit as st
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from cache import SemanticCache
from guardrails import is_prompt_injection, is_on_topic, verify_response_grounded
from rag import RAGController

load_dotenv()

st.set_page_config(page_title="GitLab Chat", page_icon="🦊", layout="wide")

st.markdown(
    "<style>[data-testid='stSidebar'] { display: none !important; }</style>",
    unsafe_allow_html=True,
)

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


def init_rag(api_key: str, supabase_url: str, supabase_key: str) -> RAGController | None:
    """Init RAGController with error handling."""
    try:
        return RAGController(
            api_key=api_key,
            supabase_url=supabase_url,
            supabase_key=supabase_key,
        )
    except Exception as e:
        return None


cache = SemanticCache()
rag = None
if st.session_state.api_key and st.session_state.supabase_url and st.session_state.supabase_key:
    rag = init_rag(
        st.session_state.api_key,
        st.session_state.supabase_url,
        st.session_state.supabase_key,
    )
    if rag is None:
        st.error("Failed to initialize RAG controller. Check your `.env` credentials.")

tab1, tab2, tab3 = st.tabs(["Chat", "Analytics", "Admin"])

with tab1:
    if not rag:
        st.markdown(
            """
            <div style="text-align: center; padding: 80px 20px;">
                <div class="welcome-icon" style="margin: 0 auto 20px;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#e85d04" stroke-width="1.5">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    </svg>
                </div>
                <div class="welcome-title">GitLab Handbook</div>
                <div class="welcome-sub">Set your API keys in <code>.env</code> to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align: center; padding: 60px 20px 40px;">
                    <div class="welcome-icon" style="margin: 0 auto 20px;">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#e85d04" stroke-width="1.5">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                        </svg>
                    </div>
                    <div class="welcome-title">GitLab Handbook</div>
                    <div class="welcome-sub">Ask anything about GitLab's policies, engineering practices, and culture.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


            chips = [
                ("PTO Policy", "What's the time off policy?"),
                ("Code Reviews", "How should I conduct code reviews?"),
                ("Career Growth", "What are the engineering levels?"),
                ("Remote Work", "What's the remote work policy?"),
            ]

            cols = st.columns(2)
            for idx, (title, desc) in enumerate(chips):
                with cols[idx % 2]:
                    with st.container():
                        st.markdown(f'<div class="suggestion-chip">', unsafe_allow_html=True)
                        if st.button(f"{title}\n\n{desc}", key=f"chip_{idx}", use_container_width=True):
                            st.session_state.last_click = desc
                        st.markdown('</div>', unsafe_allow_html=True)

        user_query = st.chat_input("Ask about the GitLab handbook...")

        query_to_process = None
        if user_query:
            query_to_process = user_query
        elif "last_click" in st.session_state:
            query_to_process = st.session_state.pop("last_click")

        if query_to_process:
            query = query_to_process

            # 1. Input Guardrails
            if is_prompt_injection(query):
                st.error("Unsafe query pattern detected. Please rephrase your question.")
            elif not is_on_topic(query):
                st.warning(
                    "I'm specialized in the GitLab Handbook. Try asking about GitLab's values, culture, hiring, or product direction."
                )
            else:
                # 2. Embed query + cache lookup
                with st.spinner("Searching..."):
                    try:
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            embed_future = executor.submit(rag.get_query_embedding, query)
                            query_embedding = embed_future.result()
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
                            # 3. Retrieval & generation
                            result = rag.query(query, query_embedding=query_embedding)
                            response_text = result["response"]

                            # 4. Async output guardrail — show response immediately, verify in background
                            msg_index = len(st.session_state.chat_history)
                            st.session_state.chat_history.append(
                                {
                                    "query": query,
                                    "response": response_text,
                                    "cache": "MISS",
                                    "latency": result["latency"],
                                    "time_to_first_token": result["time_to_first_token"],
                                    "chunks": result["retrieved_chunks"],
                                    "guardrail_status": "pending",
                                }
                            )
                            # Store in cache
                            cache.store(query, query_embedding, response_text)

                            # Use guardrail in background thread
                            def _run_guardrail(idx, resp, chunks, api_key):
                                try:
                                    is_safe = verify_response_grounded(resp, chunks, api_key)
                                    try:
                                        st.session_state.chat_history[idx]["guardrail_status"] = "safe" if is_safe else "unsafe"
                                    except Exception:
                                        pass 
                                except Exception:
                                    try:
                                        st.session_state.chat_history[idx]["guardrail_status"] = "error"
                                    except Exception:
                                        pass

                            threading.Thread(
                                target=_run_guardrail,
                                args=(msg_index, response_text, result["retrieved_chunks"], st.session_state.api_key),
                                daemon=True,
                            ).start()
                    except Exception as e:
                        error_str = str(e)
                        if "match_data_embeddings" in error_str or "PGRST202" in error_str:
                            st.error(
                                "Database setup required. The vector search function is missing. "
                                "Go to your Supabase Dashboard → SQL Editor and run the "
                                "contents of `supabase_setup.sql`."
                            )
                        else:
                            st.error(f"Error: {error_str}")

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(msg["query"])
            with st.chat_message("assistant"):
                st.markdown(msg["response"])

                # Metadata
                cache_class = "cache-hit" if msg.get("cache") == "HIT" else "cache-miss"
                cache_icon = "●" if msg.get("cache") == "HIT" else "○"
                guardrail_status = msg.get("guardrail_status", "")
                guardrail_pill = ""
                if guardrail_status == "unsafe":
                    guardrail_pill = '<span class="meta-pill" style="background:#fee2e2;color:#dc2626;">⚠ Flagged</span>'
                elif guardrail_status == "pending":
                    guardrail_pill = '<span class="meta-pill" style="background:#fef3c7;color:#d97706;">⏳ Verifying</span>'

                st.markdown(
                    f"""
                    <div class="meta-bar">
                        <span class="meta-pill {cache_class}">{cache_icon} {msg.get('cache', 'N/A')}</span>
                        <span class="meta-pill">{msg.get('latency', 0):.2f}s</span>
                        <span class="meta-pill">TTFT: {msg.get('time_to_first_token', 0):.2f}s</span>
                        {guardrail_pill}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Citations
                if msg.get("chunks"):
                    with st.expander("Sources"):
                        for idx, chunk in enumerate(msg["chunks"]):
                            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
                            source_url = meta.get("url", "https://handbook.gitlab.com/")
                            source_title = meta.get("title", "Handbook Page")
                            st.markdown(f"**{idx + 1}.** [{source_title}]({source_url})")
                            st.code(
                                chunk.get_content()[:500] + ("..." if len(chunk.get_content()) > 500 else ""),
                                language="markdown",
                            )


with tab2:
    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"<div class='metric-card'><h4>Latency</h4>"
                f"<h3>{last_msg.get('latency', 0):.3f}s</h3></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'><h4>Cache</h4>"
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
                f"<div class='metric-card'><h4>Model</h4>"
                f"<h3>{st.session_state.model_name}</h3></div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Query History")
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
        st.info("Submit a query in the Chat tab to view analytics.")

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Semantic Cache")
        cache_stats = cache.get_stats()
        st.metric("Cached Entries", cache_stats["total_entries"])

        if st.button("Clear Cache"):
            cache.clear_cache()
            st.success("Cache cleared.")
            st.rerun()

    with col2:
        st.markdown("#### Database")
        st.markdown(
            "To update the handbook index, run `python ingest.py` with your Supabase DB connection string. "
            "Embeddings are generated locally via fastembed."
        )

    st.markdown("---")
    st.markdown("#### Ingestion")
    if st.button("Run Ingestion Pipeline"):
        if not all([st.session_state.api_key, st.session_state.supabase_url, st.session_state.supabase_key]):
            st.error("Set all credentials in your `.env` file first.")
        else:
            with st.spinner("Running ingestion pipeline..."):
                import subprocess

                try:
                    db_connection = os.environ.get("SUPABASE_DB_CONNECTION", "")
                    result = subprocess.run(
                        [
                            sys.executable, "ingest.py",
                            "--db-connection", db_connection,
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
                    st.error(f"Error: {e}")
