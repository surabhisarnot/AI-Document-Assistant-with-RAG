import json
from ai_document_assistant_with_rag.retrieve import retrieve

K = 4
questions = [q for q in json.load(open("eval/questions.json")) if q.get("answerable", True)]
hits = 0
rr_sum = 0.0

for q in questions:
    chunks = retrieve(q["question"], k=K)
    snippet = q["answer_snippet"].lower()
    rank = next((i for i, c in enumerate(chunks, 1) if snippet in c["text"].lower()), None)
    if rank:
        hits += 1
        rr_sum += 1 / rank
    else:
        print("MISS:", q["question"])

n = len(questions)
print(f"Hit rate @{K}: {hits}/{n} = {hits/n:.0%}")
print(f"MRR: {rr_sum/n:.2f}")