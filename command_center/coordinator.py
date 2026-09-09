import os
import json
import anthropic
from agents import FireAgent, MedicalAgent, RouteAgent

MODEL = "claude-sonnet-4-5"

fire_agent = FireAgent()
medical_agent = MedicalAgent()
route_agent = RouteAgent()

AGENT_TOOLS = [
    {
        "name": "consult_fire_agent",
        "description": "Consult the Fire Agent for information about active fires, smoke detections, and fire-related risk scores.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "consult_medical_agent",
        "description": "Consult the Medical Agent for information about trapped people, injuries, and medical unit availability.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "consult_route_agent",
        "description": "Consult the Route Agent for information about evacuation route status and recommendations.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

AGENT_FUNCTIONS = {
    "consult_fire_agent": lambda: fire_agent.get_status(),
    "consult_medical_agent": lambda: medical_agent.get_status(),
    "consult_route_agent": lambda: route_agent.get_status(),
}

COORDINATOR_PROMPT = """You are the Decision Agent in AegisAI's multi-agent emergency command system.

You coordinate three specialist agents:
- Fire Agent: fire/smoke detection and fire risk
- Medical Agent: trapped people, injuries, medical resources
- Route Agent: evacuation routing status

Rules:
- Only consult the specialist agent(s) actually relevant to the operator's question. Don't consult all three if only one is needed.
- Synthesize their responses into a single clear, operational answer for the operator.
- If multiple agents are relevant, explain how their information relates (e.g., "the fire in Corridor A is why the route is blocked, and why 3 people are trapped nearby").
- Be concise. This is for live emergency operations, not casual conversation.
"""

# Simple keyword routing used ONLY when the LLM coordinator is unavailable.
# This is a deliberately simpler substitute for the LLM's judgment - it won't
# handle nuance or multi-topic questions as well, but keeps the multi-agent
# pattern's core behavior (selective consultation) working without an API key.
OFFLINE_ROUTING_KEYWORDS = {
    "fire": ["fire", "smoke", "burning", "flame"],
    "medical": ["trapped", "injured", "injury", "medical", "hurt", "rescue", "people need"],
    "route": ["route", "evacuat", "exit", "path", "safe to", "escape"],
}


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def offline_selective_summary(question):
    q_lower = question.lower()
    relevant_agents = []

    if any(kw in q_lower for kw in OFFLINE_ROUTING_KEYWORDS["fire"]):
        relevant_agents.append(("FIRE", fire_agent))
    if any(kw in q_lower for kw in OFFLINE_ROUTING_KEYWORDS["medical"]):
        relevant_agents.append(("MEDICAL", medical_agent))
    if any(kw in q_lower for kw in OFFLINE_ROUTING_KEYWORDS["route"]):
        relevant_agents.append(("ROUTE", route_agent))

    if not relevant_agents:
        # fall back to consulting everyone if we can't tell what's relevant
        relevant_agents = [("FIRE", fire_agent), ("MEDICAL", medical_agent), ("ROUTE", route_agent)]

    lines = ["[OFFLINE MODE - keyword-based agent routing, no LLM available]\n"]
    for name, agent in relevant_agents:
        print(f"  [Coordinator consulting: {name} AGENT]")
        lines.append(agent.report())

    return "\n".join(lines)


def ask_coordinator(question, client):
    if client is None:
        return offline_selective_summary(question)

    messages = [{"role": "user", "content": question}]

    try:
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=COORDINATOR_PROMPT,
                tools=AGENT_TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        agent_name = block.name.replace("consult_", "").replace("_agent", "").upper()
                        print(f"  [Coordinator consulting: {agent_name} AGENT]")
                        result = AGENT_FUNCTIONS[block.name]()
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                return "".join(b.text for b in response.content if b.type == "text")

    except Exception as e:
        print(f"  [LLM unavailable ({e}) - falling back to keyword-based routing]")
        return offline_selective_summary(question)


if __name__ == "__main__":
    print("AegisAI Phase 11 - Multi-Agent Command Coordinator")
    print("Ask questions. The coordinator will consult only the relevant specialist agents. Type 'quit' to exit.\n")

    client = get_client()
    if client is None:
        print("Note: No ANTHROPIC_API_KEY found. Using keyword-based offline routing.\n")

    while True:
        q = input("Operator: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        print()
        answer = ask_coordinator(q, client)
        print(f"\nCoordinator: {answer}\n")
