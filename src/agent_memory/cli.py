"""Local chat entry point: python -m agent_memory.cli chat --user alice."""

import argparse
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from agent_memory.embedder import FastEmbedEmbedder, HashEmbedder
from agent_memory.graph import build_graph
from agent_memory.llm import create_llm
from agent_memory.store import MemoryStore


def main() -> None:
    """Start a new short-term thread while retaining the user's long-term memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    chat = subcommands.add_parser("chat", help="Chat in a new thread")
    chat.add_argument("--user", required=True, help="Local user identifier")
    chat.add_argument("--db", type=Path, default=Path("data/memory.lance"))
    chat.add_argument("--embedder", choices=("fastembed", "hash"), default="fastembed")
    chat.add_argument("--offline", action="store_true", help="Require a cached FastEmbed model")
    chat.add_argument(
        "--demo", action="store_true", help="Use the deterministic offline demo model"
    )
    args = parser.parse_args()
    llm = create_llm(demo=args.demo)
    if args.embedder == "hash":
        embedder = HashEmbedder()
    else:
        embedder = FastEmbedEmbedder(local_files_only=args.offline)
    store = MemoryStore(args.db, embedder)
    graph = build_graph(store, llm)
    thread_id = f"{args.user}:{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    print("Local memory chat. Enter /quit to leave.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if not text:
            continue
        result = graph.invoke(
            {"user_id": args.user, "messages": [HumanMessage(content=text)]}, config=config
        )
        print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
