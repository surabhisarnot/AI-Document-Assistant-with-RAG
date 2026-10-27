import sys
from pathlib import Path

import chromadb

from ai_document_assistant_with_rag.ingest import load_pdf, chunk_pages
from ai_document_assistant_with_rag.store import add_chunks, DB_PATH, COLLECTION_NAME

size = int(sys.argv[1]) if len(sys.argv) > 1 else 800
overlap = int(sys.argv[2]) if len(sys.argv) > 2 else 100

client = chromadb.PersistentClient(path=DB_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass  # collection didn't exist yet

total = 0
for pdf in sorted(Path("data/sample_docs").glob("*.pdf")):
    n = add_chunks(chunk_pages(load_pdf(str(pdf)), size=size, overlap=overlap))
    print(f"  {pdf.name}: {n} chunks")
    total += n
print(f"size={size} overlap={overlap} -> {total} chunks total")