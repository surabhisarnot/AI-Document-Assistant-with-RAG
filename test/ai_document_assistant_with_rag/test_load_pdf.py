# scripts/try_load.py
from ai_document_assistant_with_rag.ingest import load_pdf

pages = load_pdf("data/sample_docs/India_History.pdf")
print(f"{len(pages)} pages loaded")
print(pages[0]["text"][:500])