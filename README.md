# ask-agent

A Google ADK agent that answers company policy questions from my own
retrieval service, with citations, and opens a ticket when the documents do
not cover the question.

Built for Session 3 of the AI Engineering Bootcamp, on top of the `/ask`
RAG service I built in Sessions 1 and 2 ([ask-rag](https://github.com/ChiaraQui/ask-rag)).

## The job in one sentence

When an employee asks a policy question, the agent answers it with document
citations, or opens a ticket when the documents do not cover it, using my
ask-rag retrieval endpoint as its tool.

## Is this an agent or a workflow?

An agent. The number of searches is not decided in advance. The agent reads
the passages and their scores, judges whether they answer the question, and
then answers, searches again with different wording, or escalates. That
branch depends on a tool result, so it cannot be hard-coded.

A fixed search-then-answer chain would be a workflow, and it would have no
way to handle a question the documents do not cover.

There is no router. The sample's Demo 1 routes between billing, technical
and escalation specialists because those roles are genuinely separate. This
job has one role, so a router would have nothing to decide.

## The one real tool

`search_policy_docs` calls `GET /debug/retrieve` on the live ask-rag service
and returns the retrieved chunks with their scores and `document_id`s. No
hardcoded data.

I used `/debug/retrieve` rather than `/ask` on purpose. It returns raw chunks
without calling an LLM, so the reasoning happens in the ADK agent where the
trace can show it, instead of being hidden inside the Session 2 service.

Tool failures come back as observations, not exceptions. The function catches
timeouts, HTTP errors, unreachable hosts and non-JSON responses, and returns
a dict with `error`, `detail` and `recoverable`. The model reads that and
decides whether to retry or escalate.

The second tool, `create_support_ticket`, is still a placeholder that returns
a canned dict.

## Bounding the loop

`RunConfig(max_llm_calls=10)` caps the loop. It fails closed: when the cap is
hit, any partial answer is discarded and `ask()` returns `stopped=True` with
a refusal telling the employee to escalate. A half-finished answer from an
agent that had not finished checking its sources is worse than no answer.

## Running it

You need a Google API key and a reachable ask-rag deployment. The agent calls
that service over HTTP, so this repo cannot run on its own.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set `GOOGLE_API_KEY` and point `ASK_API_URL` at your own deployment, then:

```bash
python policy_agent.py          # three scenarios, trace printed to stdout
streamlit run streamlit_app.py  # browser UI, "Policy Agent (my build)"
```

The Streamlit page shows each step labelled ACT, OBSERVE, THINK or STOP, and
the final answer below.

Two things worth knowing. The Gemini free tier allows 20 requests per day per
model, which is roughly ten questions, so `GEMINI_MODEL` in `.env` is there to
switch models when you run out. Render's free tier sleeps, so the first search
of the day can take around 50 seconds. The tool has a 90 second timeout.

## The course sample

`demo1_routing.py`, `demo2_mcp.py`, `demo3_full_system.py`,
`shipping_agent.py` and `week3_notebook.ipynb` come from the course sample
repo ([akshika47/AI-Internship](https://github.com/akshika47/AI-Internship/tree/main/ai-engineering-bootcamp/adk-multi-agent-systems)).
They are kept here for reference, with the fixes needed to make them run. The
commit history separates the sample, those fixes, and my own work.

`streamlit_app.py` is the sample's app with my agent added as a page.

| Demo | What it shows | Protocol |
|------|--------------|----------|
| Demo 1 | Router agent delegates to billing, technical and escalation specialists | Local tools |
| Demo 2 | Agent queries a Supabase database, tools auto-discovered at runtime | MCP |
| Demo 3 | Routing and MCP plus A2A to a remote shipping agent | MCP + A2A |

Demos 2 and 3 need Supabase credentials. Demo 3 also needs the shipping agent
running:

```bash
uvicorn shipping_agent:app --port 8001
```
