import json
from ai_document_assistant_with_rag.pipeline import ask

for a in json.load(open("eval/attacks.json", encoding="utf-8")):
    r = ask(a["question"])
    text = r["answer"]
    leaked = [s for s in a.get("must_not_contain", []) if s.lower() in text.lower()]
    refusal_ok = not a.get("expect_refusal") or r["refused"]
    print("PASS" if not leaked and refusal_ok else "FAIL", "|", a["question"][:60], "->", text[:80])