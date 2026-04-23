"""
REPL with HITL interrupt/resume loop for the multi-agent research system.
Supports --mode research (default) and --mode android.
"""
import argparse
import uuid
from langgraph.types import Command

from supervisor import build_supervisor


def _stream_supervisor(supervisor, messages, config):
    """Stream supervisor events, return (final_answer, interrupted_state)."""
    final_answer = None
    interrupted_payload = None

    for event in supervisor.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            role = getattr(last, "type", "")
            if role == "ai" and hasattr(last, "content") and last.content:
                final_answer = last.content

    # Check if graph is interrupted
    state = supervisor.get_state(config)
    if state.next:
        # Graph is paused at an interrupt
        interrupted_payload = None
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupted_payload = task.interrupts[0].value
                break

    return final_answer, interrupted_payload


def _resume_stream(supervisor, command, config):
    """Resume after interrupt, return (final_answer, next_interrupted_payload)."""
    final_answer = None

    for event in supervisor.stream(
        command,
        config=config,
        stream_mode="values",
    ):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            if hasattr(last, "type") and last.type == "ai" and last.content:
                final_answer = last.content

    state = supervisor.get_state(config)
    interrupted_payload = None
    if state.next:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupted_payload = task.interrupts[0].value
                break

    return final_answer, interrupted_payload


def handle_hitl(supervisor, payload, config):
    """Handle the HITL approval loop. Returns final answer string."""
    while payload is not None:
        filename = payload.get("filename", "report.md")
        preview = payload.get("content_preview", "")
        full_content = payload.get("full_content", "")

        print("\n" + "=" * 60)
        print("⏸️  ACTION REQUIRES APPROVAL")
        print("=" * 60)
        print(f"  Tool:  save_report")
        print(f"  File:  {filename}")
        print(f"\n--- Report Preview ---")
        print(preview)
        print("----------------------")

        while True:
            try:
                action = input("\n👉 approve / edit / reject: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                action = "reject"

            if action == "approve":
                cmd = Command(resume={"action": "approve"})
                final, payload = _resume_stream(supervisor, cmd, config)
                return final

            elif action == "edit":
                try:
                    feedback = input("✏️  Your feedback: ").strip()
                except (EOFError, KeyboardInterrupt):
                    feedback = ""
                cmd = Command(resume={"action": "edit", "feedback": feedback})
                final, payload = _resume_stream(supervisor, cmd, config)
                if payload is not None:
                    # Another interrupt (supervisor revised and called save_report again)
                    break  # re-enter outer while loop
                return final

            elif action == "reject":
                cmd = Command(resume={"action": "reject", "reason": "User rejected"})
                final, payload = _resume_stream(supervisor, cmd, config)
                return final

            else:
                print("  Please enter 'approve', 'edit', or 'reject'.")

    return None


def run_android_mode(user_story: str | None = None):
    """Run the Android development pipeline."""
    import logging
    from android_pipeline import get_android_pipeline

    # Suppress LangGraph msgpack serialization warnings for custom Pydantic types
    import os as _os
    _os.environ.setdefault("LANGGRAPH_ALLOWED_MSGPACK_MODULES", "schemas")

    # Enable INFO logging so agent tool calls and LLM messages are visible
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not user_story:
        try:
            user_story = input("Enter your Android user story: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

    if not user_story:
        print("[ERROR] No user story provided.")
        return

    pipeline = get_android_pipeline()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_story": user_story,
        "spec": None,
        "code": None,
        "review": None,
        "iteration": 0,
        "messages": [],
        "hitl_feedback": None,
        "app_name": None,
        "package_name": None,
    }

    print(f"\n🤖 Android Pipeline — user story: {user_story}")
    print("=" * 60)

    def _print_messages(events_messages: list, prefix: str = "") -> None:
        """Print new LLM/tool messages since last call."""
        for msg in events_messages:
            role = getattr(msg, "type", "")
            content = getattr(msg, "content", "")

            if role == "human":
                print(f"\n{prefix}👤 USER:\n{content}")

            elif role == "ai":
                # Tool calls
                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    # Truncate long args for readability
                    args_str = str(args)
                    if len(args_str) > 200:
                        args_str = args_str[:200] + "..."
                    print(f"\n{prefix}🔧 TOOL CALL → {name}({args_str})")
                # Text response
                if content and isinstance(content, str) and content.strip():
                    print(f"\n{prefix}🤖 LLM:\n{content[:1000]}{'...' if len(content) > 1000 else ''}")

            elif role == "tool":
                name = getattr(msg, "name", "tool")
                if isinstance(content, str) and content.strip():
                    # Always show full build output, truncate others
                    if "BUILD" in content or "FAILED" in content or "ERROR" in content:
                        print(f"\n{prefix}📤 {name} OUTPUT:\n{content}")
                    else:
                        preview = content[:500] + ("..." if len(content) > 500 else "")
                        print(f"\n{prefix}📤 {name}: {preview}")

    seen_msg_count = 0

    try:
        for event in pipeline.stream(initial_state, config=config, stream_mode="values"):
            msgs = event.get("messages", [])
            new_msgs = msgs[seen_msg_count:]
            if new_msgs:
                _print_messages(new_msgs)
                seen_msg_count = len(msgs)

            spec = event.get("spec")
            code = event.get("code")
            review = event.get("review")
            iteration = event.get("iteration", 0)

            if spec and not code:
                print(f"\n📋 BA → Spec: '{spec.title}' [{spec.estimated_complexity}]")

            if code:
                app_name = event.get("app_name") or code.app_name
                pkg = event.get("package_name") or code.package_name
                print(f"\n💻 Developer → App: '{app_name}' | Package: {pkg}")
                print(f"   Files: {len(code.files_created)} created")

            if review:
                icon = "✅" if review.verdict == "APPROVED" else "❌"
                print(f"\n{icon} QA [iter {iteration}] → {review.verdict} (score: {review.score:.2f})")
                if review.issues:
                    for issue in review.issues:
                        print(f"   ⚠️  {issue}")
                if review.suggestions:
                    for s in review.suggestions:
                        print(f"   💡 {s}")

        # Handle HITL interrupts
        state = pipeline.get_state(config)
        while state.next:
            interrupted_payload = None
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupted_payload = task.interrupts[0].value
                    break

            if interrupted_payload is None:
                break

            spec = interrupted_payload.get("spec")
            print("\n" + "=" * 60)
            print("⏸️  HITL: Spec requires your approval")
            print("=" * 60)
            if spec:
                print(f"  Title:      {spec.title}")
                print(f"  Complexity: {spec.estimated_complexity}")
                print(f"\n  Requirements:")
                for r in spec.requirements:
                    print(f"    • {r}")
                print(f"\n  Acceptance Criteria:")
                for ac in spec.acceptance_criteria:
                    print(f"    • {ac}")

            while True:
                try:
                    action = input("\n👉 approve / reject (with feedback): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    action = "reject"

                if action == "approve":
                    cmd = Command(resume={"action": "approve"})
                    break
                elif action.startswith("reject"):
                    feedback = ""
                    if ":" in action:
                        feedback = action.split(":", 1)[1].strip()
                    else:
                        try:
                            feedback = input("✏️  Rejection feedback: ").strip()
                        except (EOFError, KeyboardInterrupt):
                            pass
                    cmd = Command(resume={"action": "reject", "feedback": feedback})
                    break
                else:
                    print("  Please enter 'approve' or 'reject'.")

            for event in pipeline.stream(cmd, config=config, stream_mode="values"):
                msgs = event.get("messages", [])
                new_msgs = msgs[seen_msg_count:]
                if new_msgs:
                    _print_messages(new_msgs)
                    seen_msg_count = len(msgs)
                spec = event.get("spec")
                if spec:
                    print(f"\n📋 BA revised spec: '{spec.title}'")

            state = pipeline.get_state(config)

        # Final summary
        final = pipeline.get_state(config).values
        code = final.get("code")
        review = final.get("review")
        app_name = final.get("app_name", "")
        package_name = final.get("package_name", "")

        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        if app_name:
            print(f"  App name:    {app_name}")
        if package_name:
            print(f"  Package:     {package_name}")
        if code:
            print(f"  Output dir:  {code.files_created[0].split('/')[0:2]}")
            print(f"  Files:       {len(code.files_created)}")
        if review:
            icon = "✅" if review.verdict == "APPROVED" else "❌"
            print(f"  QA verdict:  {icon} {review.verdict} (score: {review.score:.2f})")
            if review.issues:
                print(f"  Issues:")
                for i in review.issues:
                    print(f"    • {i}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent System")
    parser.add_argument(
        "--mode",
        choices=["research", "android"],
        default="research",
        help="Pipeline mode: 'research' (default) or 'android'",
    )
    parser.add_argument(
        "user_story",
        nargs="?",
        default=None,
        help="User story for android mode (optional; will prompt if omitted)",
    )
    args = parser.parse_args()

    if args.mode == "android":
        run_android_mode(args.user_story)
        return

    # --- research mode (original behaviour) ---
    print("🤖 Multi-Agent Research System (type 'exit' to quit)")
    print("-" * 60)

    supervisor = build_supervisor()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # New conversation thread per request
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        try:
            messages = [{"role": "user", "content": user_input}]
            final_answer, interrupted_payload = _stream_supervisor(supervisor, messages, config)

            if interrupted_payload is not None:
                final_answer = handle_hitl(supervisor, interrupted_payload, config)

            if final_answer:
                print(f"\nAgent: {final_answer}")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
