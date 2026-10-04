import os
import json
import asyncio
from typing import TypedDict, List, Dict
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from rag_engine.job_matcher import match_resumes_to_job
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "filesystem": {
        "command": "python3",
        "args": ["filesystem_mcp_server.py"],
        "transport": "stdio",
    }
})

_tool_cache: dict = {}

async def get_tool(name: str):
    """Fetch an MCP tool by name, caching the tool list after first lookup."""
    if not _tool_cache:
        tools = await mcp_client.get_tools()
        _tool_cache.update({t.name: t for t in tools})
    return _tool_cache[name]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def strip_json_fence(text):
    """
    Removes markdown code fences (```json ... ``` or ``` ... ```) that LLMs
    sometimes wrap JSON responses in, despite instructions not to.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop the opening ```json or ``` line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop the closing ```
        text = "\n".join(lines)
    return text.strip()

def parse_mcp_result(raw):
    """MCP tool results may come back as a plain JSON string, a list of content-block
    objects (with a .text attribute), or a list of plain dicts (with a "text" key)."""
    if isinstance(raw, list) and raw:
        item = raw[0]
        if hasattr(item, "text"):
            raw = item.text
        elif isinstance(item, dict) and "text" in item:
            raw = item["text"]
        else:
            raw = item
    return json.loads(raw) if isinstance(raw, str) else raw


class AgentState(TypedDict):
    conversation_history: List[Dict]
    job_description: str
    requirements: Dict
    candidate_shortlist: List[Dict]
    last_user_message: str
    next_action: str

async def parse_jd(state: AgentState) -> AgentState:
    """
    Reads the JD from the last user message. If it looks like a file path, read it
    via the MCP filesystem server; otherwise treat the message itself as the JD text.
    """
    user_input = state["last_user_message"]

    if user_input.endswith(".txt"):
        read_file_tool = await get_tool("read_file")
        result = parse_mcp_result(await read_file_tool.ainvoke({"text_file": user_input}))

        if not result.get("success"):
            print(f"[parse_jd] Failed to read '{user_input}': {result.get('error')}. Treating input as raw JD text.")
            jd_text = user_input
        else:
            jd_text = result["content"]
    else:
        jd_text = user_input

    print(f"[parse_jd] Loaded JD ({len(jd_text)} chars)")
    return {"job_description": jd_text}

def extract_requirements(state: AgentState) -> AgentState:
    """
    First run: extracts requirements from the JD text.
    Subsequent runs (refinement): adjusts existing requirements based on user feedback,
    preserving any fields the user didn't ask to change.
    """
    existing_requirements = state.get("requirements", {})
    user_feedback = state.get("last_user_message", "")

    if not existing_requirements:
        prompt = f"""Analyze this job description and extract ONLY the skills/technologies explicitly 
        listed in a "REQUIRED_SKILLS" or similar skills section — not general responsibilities, domain 
        knowledge, or soft requirements mentioned in prose. Return as JSON with this exact structure:
        {{
        "must_have_skills": ["skill1", "skill2"],
        "nice_to_have_skills": ["skill1", "skill2"],
        "min_years_experience": <integer>
        }}

        Job description:
        {state['job_description']}

        Return ONLY the JSON, no other text."""
    else:
        prompt = f"""Here are the current job requirements:
{json.dumps(existing_requirements)}

The user just said: "{user_feedback}"

Update ONLY the fields the user is explicitly asking to change. Keep every other field 
EXACTLY as it currently is — do not remove or empty out skills the user didn't mention.

Return the full updated JSON with this exact structure:
{{
  "must_have_skills": [...],
  "nice_to_have_skills": [...],
  "min_years_experience": <integer>
}}

Return ONLY the JSON, no other text."""

    response = call_llm(prompt)
    response = strip_json_fence(response)

    try:
        requirements = json.loads(response)
        if existing_requirements:
            for key in ["must_have_skills", "nice_to_have_skills"]:
                if not requirements.get(key) and existing_requirements.get(key):
                    print(f"[extract_requirements] '{key}' was unexpectedly emptied — restoring previous value")
                    requirements[key] = existing_requirements[key]
    except json.JSONDecodeError:
        print(f"[extract_requirements] Failed to parse LLM response as JSON: {response}")
        requirements = existing_requirements if existing_requirements else {
            "must_have_skills": [], "nice_to_have_skills": [], "min_years_experience": 0
        }

    print(f"[extract_requirements] {requirements}")
    return {"requirements": requirements}

async def search_resumes(state: AgentState) -> AgentState:
    """
    Runs the RAG search using the JD text already stored in state.
    Writes the JD to a temp file via the MCP filesystem server, since
    match_resumes_to_job expects a filepath, not raw text.
    """
    jd_text = state["job_description"]

    write_file_tool = await get_tool("write_file")
    write_result = parse_mcp_result(
        await write_file_tool.ainvoke({"filepath": "temp_jd.txt", "content": jd_text})
    )

    if not write_result.get("success"):
        raise RuntimeError(f"[search_resumes] Failed to write temp JD file: {write_result.get('error')}")

    temp_jd_path = write_result["filepath"]  # resolved absolute path from the MCP server

    result = match_resumes_to_job(temp_jd_path, top_k=10)

    print(f"[search_resumes] Found {len(result['top_matches'])} candidates")
    return {"candidate_shortlist": result["top_matches"]}

def generate_report(state: AgentState) -> AgentState:
    requirements = state["requirements"]
    must_have = requirements.get("must_have_skills", [])
    shortlist = state["candidate_shortlist"]

    for candidate in shortlist:
        matched = candidate.get("matched_skills", [])
        missing = [s for s in must_have if s not in matched]

        prompt = f"""A candidate named {candidate['candidate_name']} scored {candidate['match_score']}/100 
        for a job requiring these must-have skills: {must_have}.
        They matched: {matched}.
        They are missing: {missing}.
        Relevant resume excerpts: {candidate['relevant_excerpts']}

        Write a short report (3-4 sentences) covering:
        1. Their key strengths for this role
        2. Any gaps versus requirements
        3. If they're borderline (score 50-75), one specific improvement suggestion

        Be concise and specific."""

        report = call_llm(prompt)

        if not report or len(report.strip()) < 30 or "safety" in report.lower()[:20]:
            print(f"[generate_report] Suspicious response for {candidate['candidate_name']}, retrying: {report!r}")
            report = call_llm(prompt)

        candidate["detailed_report"] = report
        print(f"[generate_report] Report generated for {candidate['candidate_name']}")

    return {"candidate_shortlist": shortlist}

def human_feedback(state: AgentState) -> AgentState:
    """
    Gets the next user message and classifies intent using the LLM.
    Stores the classification in next_action for the conditional edge to read.
    """
    user_input = input("\nYour turn (or type 'done' to finish): ")

    state["conversation_history"].append({"role": "user", "content": user_input})

    if user_input.strip().lower() == "done":
        return {"last_user_message": user_input, "next_action": "end"}

    prompt = f"""Classify the user's intent into exactly ONE of these categories:
    - refine_requirements: they want to change job requirements (skills, years, etc.)
    - research: they want a new/broader candidate search
    - compare: they want to compare specific candidates side by side
    - interview_questions: they want interview questions for a candidate
    - explain: they're asking why one candidate ranked higher/lower than another
    - end: they're satisfied and done

    User message: "{user_input}"

    Respond with ONLY the category name, nothing else."""

    classification = call_llm(prompt).strip().lower()

    valid_actions = ["refine_requirements", "research", "compare", "interview_questions", "explain", "end"]
    if classification not in valid_actions:
        print(f"[human_feedback] Unrecognized classification '{classification}', defaulting to 'explain'")
        classification = "explain"

    print(f"[human_feedback] Classified as: {classification}")
    return {"last_user_message": user_input, "next_action": classification}

def compare_candidates(state: AgentState) -> AgentState:
    """
    Head-to-head comparison of specific candidates the user names,
    or the top 3 if none are specified.
    """
    user_msg = state["last_user_message"]
    shortlist = state["candidate_shortlist"]

    mentioned = [c for c in shortlist if c["candidate_name"].split()[0].lower() in user_msg.lower()]
    candidates_to_compare = mentioned if len(mentioned) >= 2 else shortlist[:3]

    comparison_data = "\n\n".join([
        f"{c['candidate_name']} (score: {c['match_score']}): {c['detailed_report']}"
        for c in candidates_to_compare
    ])

    prompt = f"""Compare these candidates head-to-head for the role. Highlight key differences 
    in skills, experience, and overall fit. Be specific and concise (4-6 sentences).

{comparison_data}"""

    comparison_result = call_llm(prompt)
    print(f"[compare_candidates] Compared {len(candidates_to_compare)} candidates")

    state["conversation_history"].append({"role": "assistant", "content": comparison_result})
    print(f"\n{comparison_result}\n")

    return {"conversation_history": state["conversation_history"]}

def generate_interview_questions(state: AgentState) -> AgentState:
    """
    Generates screening questions tailored to a specific candidate's gaps/strengths.
    """
    user_msg = state["last_user_message"]
    shortlist = state["candidate_shortlist"]

    mentioned = [c for c in shortlist if c["candidate_name"].split()[0].lower() in user_msg.lower()]
    candidate = mentioned[0] if mentioned else shortlist[0]

    prompt = f"""Generate 5 screening interview questions for this candidate, based on their 
    background and the role's requirements. Include a mix of: verifying their claimed skills, 
    probing any gaps, and assessing depth of experience.

    Candidate: {candidate['candidate_name']}
    Report: {candidate['detailed_report']}
    Must-have skills for role: {state['requirements'].get('must_have_skills', [])}

Format as a numbered list."""

    questions = call_llm(prompt)
    print(f"[generate_interview_questions] Generated for {candidate['candidate_name']}")

    state["conversation_history"].append({"role": "assistant", "content": questions})
    print(f"\n{questions}\n")

    return {"conversation_history": state["conversation_history"]}

def explain_ranking(state: AgentState) -> AgentState:
    """
    Answers free-form questions about why candidates ranked the way they did,
    using the existing shortlist and reports as context.
    """
    user_msg = state["last_user_message"]
    shortlist = state["candidate_shortlist"]

    context = "\n\n".join([
        f"{c['candidate_name']} (score: {c['match_score']}, matched skills: {c['matched_skills']}): {c['detailed_report']}"
        for c in shortlist
    ])

    prompt = f"""Given this candidate shortlist and their reports, answer the user's question 
    clearly and specifically, referencing actual scores and skills.

    Candidates:
    {context}

    User's question: "{user_msg}"
    """

    answer = call_llm(prompt)
    print(f"[explain_ranking] Answered question")

    state["conversation_history"].append({"role": "assistant", "content": answer})
    print(f"\n{answer}\n")

    return {"conversation_history": state["conversation_history"]}

def final_recommendation(state: AgentState) -> AgentState:
    """
    Round 3 of screening: produces a definitive hire/no-hire recommendation
    for each top candidate, based on their score and detailed report.
    """
    shortlist = state["candidate_shortlist"]
    requirements = state["requirements"]

    top_candidates = shortlist[:5]

    for candidate in top_candidates:
        prompt = f"""Based on this candidate's evaluation, give a final hiring recommendation.

    Candidate: {candidate['candidate_name']}
    Match score: {candidate['match_score']}/100
    Matched must-have skills: {candidate['matched_skills']}
    Required skills: {requirements.get('must_have_skills', [])}
    Detailed report: {candidate['detailed_report']}

    Respond with a JSON object in exactly this structure:
    {{
    "recommendation": "hire" or "no-hire" or "hire with reservations",
    "confidence": "high" or "medium" or "low",
    "justification": "1-2 sentence reason"
    }}

    Return ONLY the JSON."""

        response = call_llm(prompt)

        # NOTE: fixed a pre-existing bug — the retry call was unindented, so it
        # ran unconditionally on every candidate instead of only on a bad response.
        if not response or len(response.strip()) < 15 or "safety" in response.lower()[:20]:
            print(f"[final_recommendation] Suspicious response for {candidate['candidate_name']}, retrying: {response!r}")
            response = call_llm(prompt)

        try:
            verdict = json.loads(strip_json_fence(response))
        except json.JSONDecodeError:
            print(f"[final_recommendation] Failed to parse for {candidate['candidate_name']}: {response}")
            verdict = {"recommendation": "unknown", "confidence": "low", "justification": "Could not generate recommendation"}

        candidate["final_verdict"] = verdict
        print(f"[final_recommendation] {candidate['candidate_name']}: {verdict['recommendation']} ({verdict['confidence']})")

    return {"candidate_shortlist": shortlist}

def route_after_feedback(state: AgentState) -> str:
    return state["next_action"]

graph = StateGraph(AgentState)

graph.add_node("parse_jd", parse_jd)
graph.add_node("extract_requirements", extract_requirements)
graph.add_node("search_resumes", search_resumes)
graph.add_node("generate_report", generate_report)
graph.add_node("human_feedback", human_feedback)
graph.add_node("compare", compare_candidates)
graph.add_node("interview_questions", generate_interview_questions)
graph.add_node("explain", explain_ranking)
graph.add_node("final_recommendation", final_recommendation)
graph.add_edge(START, "parse_jd")
graph.add_edge("parse_jd", "extract_requirements")
graph.add_edge("extract_requirements", "search_resumes")
graph.add_edge("search_resumes", "generate_report")
graph.add_edge("generate_report", "final_recommendation")
graph.add_edge("final_recommendation", "human_feedback")

graph.add_conditional_edges(
    "human_feedback",
    route_after_feedback,
    {
        "refine_requirements": "extract_requirements",
        "research": "search_resumes",
        "compare": "compare",
        "interview_questions": "interview_questions",
        "explain": "explain",
        "end": END
    }
)

graph.add_edge("compare", "human_feedback")
graph.add_edge("interview_questions", "human_feedback")
graph.add_edge("explain", "human_feedback")

app = graph.compile()

async def main():
    initial_state = {
        "conversation_history": [],
        "job_description": "",
        "requirements": {},
        "candidate_shortlist": [],
        "last_user_message": "jd_01.txt",  # must exist inside the MCP server's allowed root (resumes/)
        "next_action": ""
    }

    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 50})

    print("\nTop 5 candidates with final recommendations:")
    for c in final_state["candidate_shortlist"][:5]:
        verdict = c.get("final_verdict", {})
        print(f"\n{c['candidate_name']} ({c['match_score']}/100)")
        print(f"Verdict: {verdict.get('recommendation')} (confidence: {verdict.get('confidence')})")
        print(f"Reason: {verdict.get('justification')}")

    print("\n--- Conversation ended ---")

if __name__ == "__main__":
    asyncio.run(main())