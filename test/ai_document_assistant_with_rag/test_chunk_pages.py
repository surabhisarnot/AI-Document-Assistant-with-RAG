# scripts/try_chunking.py
from ai_document_assistant_with_rag.ingest import load_pdf, chunk_pages

pages = load_pdf("data/sample_docs/RAG_Reference_Paper.pdf")
chunks = chunk_pages(pages, size=800, overlap=100)

print(f"{len(pages)} pages -> {len(chunks)} chunks")
for c in chunks[:3]:
    print("-" * 40)
    print(c["chunk_id"], "| chars:", len(c["text"]))
    print(c["text"])