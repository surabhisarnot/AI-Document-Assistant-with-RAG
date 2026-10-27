import json
from collections import defaultdict

from ai_document_assistant_with_rag.retrieve import retrieve
from ai_document_assistant_with_rag.generate import answer

K = 4
questions = json.load(open("eval/questions.json", encoding="utf-8"))

stats = defaultdict(lambda: {"ok": 0, "total": 0, "dists": []})
failures = []

for q in questions:
    chunks = retrieve(q["question"], k=K)
    result = answer(q["question"], chunks)
    refused = result["refused"]

    if q["answerable"]:
        group, ok = "answerable", not refused
    else:
        group, ok = q.get("category", "unanswerable"), refused

    s = stats[group]
    s["total"] += 1
    s["ok"] += ok
    s["dists"].append(chunks[0]["distance"] if chunks else 1.0)
    if not ok:
        failures.append((group, q["question"], result["answer"][:120]))

print("\nResults (correct behaviour per group):")
for g, s in stats.items():
    print(f"  {g:15} {s['ok']}/{s['total']} | best distance min {min(s['dists']):.3f}, max {max(s['dists']):.3f}")

if failures:
    print("\nFailures:")
    for g, question, ans in failures:
        print(f"  [{g}] {question}\n      -> {ans}")