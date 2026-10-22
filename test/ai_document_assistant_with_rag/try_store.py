from ai_document_assistant_with_rag.ingest import load_pdf, chunk_pages
from ai_document_assistant_with_rag.store import add_chunks, get_collection, search
from ai_document_assistant_with_rag.embed import embed_query

pages = load_pdf("data/sample_docs/History_of_India_Small.pdf")
chunks = chunk_pages(pages, size=800, overlap=100)

print("added:", add_chunks(chunks))
print("collection count:", get_collection().count())

question = "Who was the first Prime Minister of India?"
for r in search(embed_query(question), k=3):
    print("-" * 40)
    print(f"{r['source']} p{r['page']} | distance: {r['distance']:.3f}")
    print(r["text"][:300])