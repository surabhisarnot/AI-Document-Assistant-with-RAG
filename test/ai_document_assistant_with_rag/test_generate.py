from ai_document_assistant_with_rag.retrieve import retrieve
from ai_document_assistant_with_rag.generate import answer, build_prompt

q = "What is Indian National Congress? Was that the first governing political party of India?"
chunks = retrieve(q, k=4)

print(build_prompt(q, chunks)[:500])   # see exactly what Claude receives
print("=" * 60)

result = answer(q, chunks)
print(result["answer"])
print("sources:", result["sources"])
print("tokens in/out:", result["input_tokens"], result["output_tokens"])