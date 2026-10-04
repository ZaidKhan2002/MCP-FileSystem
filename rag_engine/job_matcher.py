import os
import re
import cohere
import chromadb
from file_assistant.fs_tool import read_file

co = cohere.Client(os.environ.get("COHERE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="resumes")


def embed_query(text):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]


def semantic_search(job_description_text, n_pool=40):
    """
    Embeds the JD and retrieves a larger pool of chunks to later group by resume.
    """
    query_embedding = embed_query(job_description_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_pool
    )

    return results


def group_by_resume(results):
    """
    Collapses chunk-level results into resume-level results.
    Keeps the best (lowest) distance per resume, and collects which sections matched.
    """
    grouped = {}

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        filepath = metadata["filepath"]
        distance = results["distances"][0][i]
        section = metadata["section"]
        text = results["documents"][0][i]

        if filepath not in grouped:
            grouped[filepath] = {
                "filepath": filepath,
                "name": metadata["name"],
                "skills": metadata["skills"],
                "years_experience": metadata["years_experience"],
                "education": metadata["education"],
                "best_distance": distance,
                "matched_sections": [section],
                "excerpts": {section: text},
            }
        else:
            entry = grouped[filepath]
            entry["matched_sections"].append(section)
            entry["excerpts"][section] = text
            if distance < entry["best_distance"]:
                entry["best_distance"] = distance

    # sort resumes by their best distance (lower = more similar)
    sorted_resumes = sorted(grouped.values(), key=lambda r: r["best_distance"])
    return sorted_resumes

def extract_must_have_skills(jd_text):
    other_headers = ["TITLE", "COMPANY", "EXPERIENCE_REQUIRED", "RESPONSIBILITIES", "QUALIFICATIONS"]
    lines = jd_text.split("\n")
    skills_text = ""
    capturing = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("REQUIRED_SKILLS"):
            capturing = True
            # handle "REQUIRED_SKILLS: Python, SQL, AWS" on one line
            if ":" in stripped:
                skills_text += stripped.split(":", 1)[1] + " "
            continue

        if any(stripped.startswith(h) for h in other_headers):
            capturing = False

        if capturing and stripped:
            skills_text += stripped + " "

    skills = [s.strip() for s in skills_text.split(",") if s.strip()]
    return skills

def extract_min_years_required(jd_text):
    """
    Looks for a pattern like '5+ years' anywhere in the JD text.
    Returns an integer, or 0 if not found.
    """
    match = re.search(r"(\d+)\+?\s+years?", jd_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def keyword_score(resume_skills_text, must_have_skills):
    """
    Checks how many must-have skills appear (case-insensitive) in the resume's SKILLS text.
    Returns (score 0-100, list of matched skill names).
    """
    if not must_have_skills:
        return 100, []  # no requirements specified = auto-pass

    resume_skills_lower = resume_skills_text.lower()
    matched = [skill for skill in must_have_skills if skill.lower() in resume_skills_lower]

    score = (len(matched) / len(must_have_skills)) * 100
    return score, matched


def distance_to_semantic_score(distance):
    """
    Converts ChromaDB's cosine distance (0=identical, ~2=opposite) into a 0-100 score.
    """
    similarity = 1 - (distance / 2)
    score = max(0, min(100, similarity * 100))
    return score


def score_resume(resume, must_have_skills):
    """
    Combines semantic score + keyword score into a final 0-100 match_score.
    Weighted 60% semantic, 40% keyword — semantic captures overall relevance,
    keyword ensures hard requirements aren't ignored.
    """
    semantic = distance_to_semantic_score(resume["best_distance"])
    keyword, matched_skills = keyword_score(resume["skills"], must_have_skills)

    final_score = round((semantic * 0.6) + (keyword * 0.4))

    return final_score, matched_skills


def build_reasoning(resume, matched_skills, final_score):
    """
    Builds a short human-readable explanation string.
    """
    sections = ", ".join(resume["matched_sections"])
    if matched_skills:
        skill_str = ", ".join(matched_skills)
        return f"Matched on {sections} sections with a score of {final_score}/100. Key matching skills: {skill_str}."
    else:
        return f"Matched on {sections} sections with a score of {final_score}/100. No must-have skills explicitly matched."


def match_resumes_to_job(jd_filepath, top_k=10):
    """
    Full pipeline: load JD -> semantic search -> group by resume ->
    score each -> sort -> return final JSON-ready dict.
    """
    jd_result = read_file(jd_filepath)
    jd_text = jd_result["content"]

    must_have_skills = extract_must_have_skills(jd_text)

    raw_results = semantic_search(jd_text, n_pool=40)
    ranked_resumes = group_by_resume(raw_results)
    min_years = extract_min_years_required(jd_text)
    ranked_resumes = [r for r in ranked_resumes if r["years_experience"] >= min_years]

    scored_matches = []
    for resume in ranked_resumes:
        final_score, matched_skills = score_resume(resume, must_have_skills)
        reasoning = build_reasoning(resume, matched_skills, final_score)

        # relevant_excerpts: pull actual text from the sections that matched
        excerpts = [resume["excerpts"][section][:200] for section in resume["matched_sections"]]

        scored_matches.append({
            "candidate_name": resume["name"],
            "resume_path": resume["filepath"],
            "match_score": final_score,
            "matched_skills": matched_skills,
            "relevant_excerpts": excerpts,
            "reasoning": reasoning
        })

    # sort by final blended score, descending (best match first)
    scored_matches.sort(key=lambda m: m["match_score"], reverse=True)

    return {
        "job_description": jd_text,
        "top_matches": scored_matches[:top_k]
    }

if __name__ == "__main__":
    import json

    output = match_resumes_to_job("job_descriptions/jd_01.txt", top_k=10)
    print(json.dumps(output, indent=2))