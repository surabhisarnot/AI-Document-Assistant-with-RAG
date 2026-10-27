import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st
import traceback

from ai_document_assistant_with_rag.guardrails import (
    GLOBAL_LIMITER, MAX_CHUNKS, RateLimiter, check_question, validate_pdf)
from ai_document_assistant_with_rag.ingest import chunk_pages, load_pdf
from ai_document_assistant_with_rag.generate import ask
from ai_document_assistant_with_rag.store import (
    COLLECTION_NAME, add_chunks, delete_collection, get_collection)

st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
st.title("📄 AI Document Assistant")
st.caption("Answers come only from the documents and cite their sources. "
           "This is a demo, so please don't upload sensitive files.")


@st.cache_resource
def ensure_samples() -> bool:
    if get_collection().count() == 0:
        for pdf in sorted(Path("data/sample_docs").glob("*.pdf")):
            add_chunks(chunk_pages(load_pdf(str(pdf)), size=800, overlap=100))
    return True


ensure_samples()

if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:12]
    st.session_state.messages = []
    st.session_state.files = []
    st.session_state.limiter = RateLimiter(max_calls=15, per_seconds=3600)

user_collection = f"session_{st.session_state.sid}"


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**[{s['id']}] {s['source']}, page {s['page']}**")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


with st.sidebar:
    mode = st.radio("Documents", ["Sample documents", "Upload my own"])
    collection = COLLECTION_NAME
    if mode == "Upload my own":
        collection = user_collection
        up = st.file_uploader("Upload a PDF", type="pdf")
        if up and up.name not in st.session_state.files:
            ok, msg = validate_pdf(up.getvalue(), up.name, len(st.session_state.files))
            if not ok:
                st.error(msg)
            else:
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", up.name)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / safe
                    path.write_bytes(up.getvalue())
                    with st.spinner("Indexing..."):
                        chunks = chunk_pages(load_pdf(str(path)), size=800, overlap=100)
                        if len(chunks) > MAX_CHUNKS:
                            st.error("Document is too large for this demo.")
                        else:
                            add_chunks(chunks, collection_name=user_collection)
                            st.session_state.files.append(up.name)
                            st.success(f"Indexed {up.name} ({len(chunks)} chunks)")
        if st.session_state.files:
            st.write("Your files:", ", ".join(st.session_state.files))
            if st.button("Delete my documents"):
                delete_collection(user_collection)
                st.session_state.files = []
                st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Ask a question about the documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        ok, msg = check_question(question)
        if not ok:
            reply, sources = msg, []
        elif mode == "Upload my own" and not st.session_state.files:
            reply, sources = "Please upload a PDF first.", []
        elif not st.session_state.limiter.allow() or not GLOBAL_LIMITER.allow():
            reply, sources = "The demo's question limit has been reached. Please try again later.", []
        else:
            try:
                with st.spinner("Thinking..."):
                    result = ask(question, collection_name=collection)
                reply, sources = result["answer"], result["sources"]
            except Exception:
                traceback.print_exc()   # full error goes to the terminal
                reply, sources = "Something went wrong. Please try again in a moment.", []
        st.markdown(reply)
        if sources:
            show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})