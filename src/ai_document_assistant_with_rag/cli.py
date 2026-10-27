import argparse

from .ingest import load_pdf, chunk_pages
from .store import add_chunks
from .retrieve import retrieve
from .generate import answer


def cmd_ingest(path: str) -> None:
    pages = load_pdf(path)
    chunks = chunk_pages(pages, size=800, overlap=100)
    n = add_chunks(chunks)
    print(f"Ingested {path}: {len(pages)} pages -> {n} chunks")


def cmd_ask(question: str, k: int) -> None:
    result = ask(question, k=k)
    if result.get("reason") == "no_documents":
        print("Nothing indexed yet. Run: rag ingest <file.pdf>")
        return
    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['source']}, page {s['page']}")
    print(f"\n(tokens in/out: {result['input_tokens']}/{result['output_tokens']})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index a PDF")
    p_ingest.add_argument("path")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4, help="chunks to retrieve")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args.path)
    else:
        cmd_ask(args.question, args.k)


if __name__ == "__main__":
    main()