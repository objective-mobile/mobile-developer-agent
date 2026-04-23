"""
Non-interactive test: runs the full Plan -> Research -> Critique pipeline
and auto-approves the save_report step.
"""
import uuid
import time
import re
from langgraph.types import Command

from supervisor import build_supervisor


def invoke_with_retry(supervisor, input_data, config, max_retries=5):
    """Invoke supervisor with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            result = supervisor.invoke(input_data, config=config)
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60
                m = re.search(r"retry in (\d+)", err)
                if m:
                    wait = int(m.group(1)) + 5
                print(f"[Supervisor] Rate limit, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for supervisor")


def run_test():
    print("=" * 60)
    print("TEST: Multi-Agent Research Pipeline")
    print("=" * 60)

    supervisor = build_supervisor()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    query = "What is Retrieval-Augmented Generation (RAG)? Write a short report."
    print(f"\nQuery: {query}\n")

    result = invoke_with_retry(supervisor, {"messages": [{"role": "user", "content": query}]}, config)

    # Check for interrupt
    state = supervisor.get_state(config)
    interrupted_payload = None
    if state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupted_payload = task.interrupts[0].value
                break

    if interrupted_payload:
        print("\nHITL interrupt triggered correctly")
        print(f"  File: {interrupted_payload.get('filename')}")
        print(f"  Preview: {interrupted_payload.get('content_preview', '')[:200]}...")
        print("\n  Auto-approving for test...")

        result = invoke_with_retry(supervisor, Command(resume={"action": "approve"}), config)
        print("\nReport saved successfully")
    else:
        print("\nNo HITL interrupt detected")

    # Get final message
    msgs = result.get("messages", [])
    final_answer = None
    for msg in reversed(msgs):
        if hasattr(msg, "type") and msg.type == "ai" and msg.content:
            final_answer = msg.content
            break

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(final_answer or "(no final answer)")
    print("\nTest complete")


if __name__ == "__main__":
    run_test()
