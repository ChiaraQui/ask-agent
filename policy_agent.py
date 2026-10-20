"""
Policy Agent -- Session 3, step 2: minimal ADK agent, placeholder tools.

THE JOB (one sentence):
    When an employee asks a policy question, the agent answers it with
    document citations -- or opens a ticket when the documents don't cover
    it -- using my capstone's retrieval endpoint as its tool.

Patterns copied from the sample repo:
    demo1_routing.py:20-57   tool functions: plain Python, type-hinted,
                             one-line docstring (ADK builds the tool schema
                             from the signature + docstring)
    demo1_routing.py:60-65   Agent(name=, model=, description=,
                             instruction=, tools=[...])
    demo1_routing.py:88-97   async ask() runner loop over run_async events
    streamlit_app.py:146-161 reading function_call / function_response off
                             event.content.parts to build a trace

Single agent on purpose -- this job has one specialist, so there is nothing
to route between. No sub_agents.

TOOLS
    search_policy_docs   REAL. GET {ASK_API_URL}/debug/retrieve -- Pinecone
                         retrieval over the six Northwind documents ingested
                         in Session 2. Returns live chunks with scores and
                         document_ids. Failures come back as observations
                         with an 'error' field, never as exceptions, so the
                         model can read them and decide what to do.
    create_support_ticket  Still a placeholder. The brief was exactly one
                         real tool, so this stays canned for now.

Run: python policy_agent.py
"""

import asyncio
import os

import requests
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# The Session 2 RAG service -- this agent's one real tool. Set ASK_API_URL in
# .env to your own deployment. Render's free tier sleeps, so the first request
# of the day wakes it and can take ~50s.
ASK_API_URL = os.getenv("ASK_API_URL", "https://your-service.onrender.com").rstrip("/")
REQUEST_TIMEOUT = 90

# Step limit: the hard stop on the Think -> Act -> Observe loop. One LLM call
# per turn, so 10 allows the three searches the instruction permits, the tool
# results coming back, and the final answer, with room to spare. Unbounded
# agents burn tokens and money.
#
# This FAILS CLOSED: when the cap is hit, ask() throws away whatever partial
# answer had accumulated and returns stopped=True. A half-finished answer is
# never passed off as a real one.
MAX_LLM_CALLS = 10

# --- Tools (PLACEHOLDERS -- real retrieval arrives in step 3) ---


def search_policy_docs(query: str, top_k: int) -> dict:
    """Search the company's Northwind policy documents and return the passages
    that best match the query, each with a relevance score and a document_id to
    cite. Call this for any question about company policy, benefits, expenses,
    security, IT use or facilities.

    Scores are cosine similarity and run low in absolute terms -- judge a
    passage by whether its text answers the question, not by the score alone.

    Args:
        query: What to search for. Use wording likely to appear in a policy
            document, not the employee's conversational phrasing.
        top_k: How many passages to return, 1 to 20. Five is sensible.
    """
    # REAL TOOL. Hits the Session 2 RAG service: Pinecone retrieval over the
    # six ingested Northwind documents. No LLM on that side -- it returns raw
    # chunks, so the reasoning stays here where the trace can show it.
    url = f"{ASK_API_URL}/debug/retrieve"
    try:
        response = requests.get(
            url,
            params={"q": query, "k": max(1, min(top_k, 20))},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        # Errors come back as observations, not exceptions. The model reads
        # them and decides what to do -- retry, or escalate.
        return {
            "error": "timeout",
            "detail": f"{url} did not respond within {REQUEST_TIMEOUT}s.",
            "recoverable": True,
            "passages": [],
        }
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        return {
            "error": "http_error",
            "detail": f"{url} returned HTTP {status}.",
            "recoverable": status in (429, 500, 502, 503, 504),
            "passages": [],
        }
    except requests.RequestException as exc:
        return {
            "error": "unreachable",
            "detail": f"Could not reach {url}: {exc}",
            "recoverable": False,
            "passages": [],
        }
    except ValueError as exc:
        return {
            "error": "bad_response",
            "detail": f"{url} returned something that is not JSON: {exc}",
            "recoverable": False,
            "passages": [],
        }

    passages = [
        {
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("id"),
            "source": chunk.get("source"),
            "score": chunk.get("score"),
            "text": chunk.get("text", ""),
        }
        for chunk in payload.get("results", [])
    ]
    return {"query": query, "passage_count": len(passages), "passages": passages}


def create_support_ticket(question: str, reason: str) -> dict:
    """Open a ticket for a human when the policy documents do not answer the
    question. Only call this after searching and finding nothing relevant.

    Args:
        question: The employee's original question, unchanged.
        reason: Why the documents could not answer it. Name the searches
            tried and what came back.
    """
    # PLACEHOLDER. A real version would POST to a ticketing system.
    return {
        "_placeholder": True,
        "ticket_id": "POL-ESC-0001",
        "status": "open",
        "question": question,
        "reason": reason,
        "estimated_response": "1 business day",
    }


# --- Agent ---

policy_agent = Agent(
    name="policy_agent",
    model=MODEL,
    description="Answers employee policy questions from the company document "
    "store, with citations.",
    instruction=(
        # GOAL
        "You answer employee questions about company policy, grounded in the "
        "company's own policy documents.\n"
        "\n"
        # CONSTRAINTS
        "Constraints:\n"
        "- Always call search_policy_docs before answering. Never answer from "
        "memory, even if you think you know the answer.\n"
        "- Only state things the retrieved passages support. Quote figures, "
        "dates and limits exactly as written. Never guess or fill gaps.\n"
        "- Search at most three times. Policy documents use formal language, "
        "so if a search misses, rephrase towards it and try again.\n"
        "- If a search comes back with an 'error' field, read it. When "
        "'recoverable' is true you may try the search once more; when it is "
        "false, do not keep retrying -- open a ticket and say the search "
        "service failed.\n"
        "- If the documents genuinely do not cover the question, call "
        "create_support_ticket instead of answering.\n"
        "\n"
        # DONE
        "You are done when one of two things is true:\n"
        "1. You have given an answer supported by retrieved passages, ending "
        "with the list of document_ids you relied on.\n"
        "2. You have opened a ticket and told the employee it is open, naming "
        "what you searched for.\n"
        "Anything else is not done. Do not stop halfway with a partial answer."
    ),
    tools=[search_policy_docs, create_support_ticket],
)

# ADK's conventional name for the entry agent; `adk web` looks for this.
root_agent = policy_agent

# --- Runner ---


async def ask(agent, message, verbose=True):
    """Run one question through the agent, printing the Think/Act/Observe loop.

    Returns (final_answer, trace, stopped). `stopped` is True when the step cap
    fired -- in that case final_answer is a refusal, not an answer. Callers must
    check it rather than string-matching the text.
    """
    service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="policy", session_service=service)
    session = await service.create_session(app_name="policy", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    trace, final = [], "(no response)"
    step = 0
    stopped = False

    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(max_llm_calls=MAX_LLM_CALLS),
        ):
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                call = getattr(part, "function_call", None)
                result = getattr(part, "function_response", None)
                text = getattr(part, "text", None)

                if call:
                    step += 1
                    args = dict(call.args) if call.args else {}
                    trace.append({"step": step, "phase": "ACT", "tool": call.name, "args": args})
                    if verbose:
                        print(f"  [{step}] ACT      call {call.name}({args})")

                elif result:
                    step += 1
                    body = str(result.response)
                    trace.append({"step": step, "phase": "OBSERVE", "tool": result.name, "result": body})
                    if verbose:
                        print(f"  [{step}] OBSERVE  {result.name} returned {body[:140]}...")

                elif text and text.strip():
                    step += 1
                    phase = "ANSWER" if event.is_final_response() else "THINK"
                    trace.append({"step": step, "phase": phase, "text": text})
                    if verbose and phase == "THINK":
                        print(f"  [{step}] THINK    {text.strip()[:140]}")
                    if event.is_final_response():
                        final = text

    except LlmCallsLimitExceededError:
        # Fail closed. Discard any partial answer: it was produced mid-loop by
        # an agent that had not finished checking its sources, so shipping it
        # would be worse than shipping nothing.
        stopped = True
        step += 1
        trace.append({"step": step, "phase": "STOP", "reason": "step_cap", "cap": MAX_LLM_CALLS})
        final = (
            f"Stopped without answering: hit the {MAX_LLM_CALLS}-step cap. "
            "No answer is given because the agent had not finished checking "
            "its sources. Escalate this question to a human."
        )
        if verbose:
            print(f"  [{step}] STOP     step cap of {MAX_LLM_CALLS} LLM calls hit -- failing closed")

    return final, trace, stopped


async def main():
    scenarios = [
        ("ANSWERABLE", "How many days a week can I work from home?"),
        ("NEEDS REPHRASING", "What do I get back if I drive my own car to a client?"),
        ("NOT IN THE DOCS", "What is the company policy on pet insurance?"),
    ]
    print(f"model={MODEL}  max_llm_calls={MAX_LLM_CALLS}  search_policy_docs -> {ASK_API_URL}")
    for label, query in scenarios:
        print(f"\n{'=' * 72}\n--- {label} ---\nUser: {query}\n")
        answer, _, stopped = await ask(policy_agent, query)
        print(f"\n{'STOPPED' if stopped else 'Agent'}: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
