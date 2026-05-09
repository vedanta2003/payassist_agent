"""Interactive CLI for the Agent. Run with: python cli.py"""
import logging
from agent import Agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

def main():
    agent = Agent()
    print("PayAssist — type 'exit' to quit.\n")
    # Opening turn
    out = agent.next("")
    print(f"Agent: {out['message']}\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        out = agent.next(user)
        print(f"\nAgent: {out['message']}\n")

if __name__ == "__main__":
    main()
