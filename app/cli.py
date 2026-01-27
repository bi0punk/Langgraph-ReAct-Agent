from app.graph import agent

print("\nRAG Agent CPU — type 'exit' to quit")

while True:
    q = input("\n> ")
    if q.lower() in ["exit", "quit"]:
        break

    result = agent.invoke({"question": q})
    print("\nAnswer:\n", result["answer"])
