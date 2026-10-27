from ai_document_assistant_with_rag.retrieve import retrieve

questions = [
    "Who was the first Prime Minister of India?",
    "National Congress Party was founded in which year?",
    "Which was the first govering polictical party of India?",
    "Whom did India get independence from?",
    "What is the capital of France?",  # not in the PDF
]

for q in questions:
    print("=" * 60)
    print("Q:", q)
    for r in retrieve(q, k=6):
        print(f"  {r['source']} p{r['page']} | distance {r['distance']:.3f}")
        print("   ", r["text"][:150].replace("\n", " "))