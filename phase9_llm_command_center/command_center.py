import os
import json
import anthropic
from system_state import get_system_status, get_active_incidents, get_recent_reports, get_evacuation_status

MODEL = "claude-sonnet-4-5"

TOOLS = [
    {
        "name": "get_system_status",
        "description": "Get the current overall system risk score, risk level, and trend.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_active_incidents",
        "description": "Get a list of currently active detected incidents (fires, trapped people, etc.) with location and confidence.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_recent_reports",
        "description": "Get recently submitted and NLP-parsed emergency reports, including extracted event type, severity, and people count.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_evacuation_status",
        "description": "Get the current evacuation routing status, including whether the primary route is blocked and what the recommended alternate route is.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_FUNCTIONS = {
    "get_system_status": get_system_status,
    "get_active_incidents": get_active_incidents,
    "get_recent_reports": get_recent_reports,
    "get_evacuation_status": get_evacuation_status,
}

SYSTEM_PROMPT = """You are the AegisAI Command Center Assistant, an AI assistant helping emergency operators understand the current situation during an active incident.

Rules:
- ONLY answer based on data retrieved from the available tools. Never make up incident details, locations, or numbers.
- If the tools don't have information to answer a question, say so clearly rather than guessing.
- Be concise and operational in tone - this is for someone actively managing an emergency, not a casual chat.
- When relevant, proactively mention risk level and recommended actions.
"""


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def offline_fallback_summary():
    """Rule-based summary used when the LLM API is unavailable (no key, no credits, no internet)."""
    status = get_system_status()
    incidents = get_active_incidents()
    evac = get_evacuation_status()

    lines = []
    lines.append("[OFFLINE MODE - LLM unavailable, showing raw system data instead]\n")
    lines.append(f"Overall risk: {status['overall_risk_score']} ({status['overall_risk_level']}), trend: {status['trend']}")

    if incidents:
        lines.append(f"\nActive incidents ({len(incidents)}):")
        for inc in incidents:
            lines.append(f"  - {inc['type']} at {inc['location']} (confidence {inc['confidence']}, status: {inc['status']})")
    else:
        lines.append("\nNo active incidents.")

    if evac.get("primary_route_blocked"):
        lines.append(f"\nEvacuation: primary route blocked. Recommended: {evac['recommended_route']} (~{evac['estimated_evacuation_time_minutes']} min)")
    else:
        lines.append("\nEvacuation: primary route clear.")

    return "\n".join(lines)


def ask_command_center(user_question, client):
    if client is None:
        return offline_fallback_summary()

    messages = [{"role": "user", "content": user_question}]

    try:
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        print(f"  [calling tool: {tool_name}]")
                        result = TOOL_FUNCTIONS[tool_name]()
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = "".join(block.text for block in response.content if block.type == "text")
                return final_text

    except anthropic.APIStatusError as e:
        if e.status_code == 400 and "credit balance" in str(e).lower():
            print("  [LLM API unavailable: insufficient credit balance - falling back to offline summary]")
        else:
            print(f"  [LLM API error ({e.status_code}) - falling back to offline summary]")
        return offline_fallback_summary()

    except anthropic.APIConnectionError:
        print("  [LLM API unreachable (no internet or connection issue) - falling back to offline summary]")
        return offline_fallback_summary()

    except Exception as e:
        print(f"  [Unexpected error: {e} - falling back to offline summary]")
        return offline_fallback_summary()


if __name__ == "__main__":
    print("AegisAI Phase 9 - LLM Command Center")
    print("Ask questions about the current emergency situation. Type 'quit' to exit.\n")

    client = get_client()
    if client is None:
        print("Note: No ANTHROPIC_API_KEY found in environment. Running in offline mode only.\n")

    while True:
        question = input("Operator: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer = ask_command_center(question, client)
        print(f"\nAegisAI: {answer}\n")
