import os
import re
import cohere
import chromadb
from file_assistant.fs_tool import read_file, list_files

co = cohere.Client(os.environ.get("COHERE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="resumes")

SECTION_HEADERS = ["NAME", "CONTACT", "SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION"]

def chunk_resume(text):
    """
    Splits resume text into a dict like:
    {"NAME": "...", "CONTACT": "...", "SUMMARY": "...", ...}
    """
    chunks = {}
    current_section = None
    current_lines = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped in SECTION_HEADERS:
            # save the previous section before starting a new one
            if current_section:
                chunks[current_section] = "\n".join(current_lines).strip()
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # save the last section after the loop ends
    if current_section:
        chunks[current_section] = "\n".join(current_lines).strip()

    return chunks

def extract_years_experience(chunks):
    """
    Looks for a pattern like '3 years' or 'Fresh graduate' in the SUMMARY section.
    Returns an integer (0 if fresh graduate / not found).
    """
    summary = chunks.get("SUMMARY", "") + " " + chunks.get("NAME", "")
    full_text_to_search = chunks.get("SUMMARY", "")

    match = re.search(r"(\d+)\+?\s+years?", full_text_to_search, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0  # fresh graduate or not mentioned


def extract_metadata(chunks):
    """
    Builds a clean metadata dict from chunked sections.
    """
    return {
        "name": chunks.get("NAME", "Unknown").strip(),
        "skills": chunks.get("SKILLS", "").strip(),
        "years_experience": extract_years_experience(chunks),
        "education": chunks.get("EDUCATION", "").strip(),
    }

def load_and_process_resumes(directory="resumes"):
    """
    Loads every .txt resume in the directory, chunks it, and extracts metadata.
    Returns a list of dicts: [{"filepath":..., "chunks":..., "metadata":...}, ...]
    """
    list_result = list_files(directory, extension=".txt")

    if not list_result["success"]:
        print(f"Error listing directory: {list_result['error']}")
        return []

    processed = []

    for file_info in list_result["files"]:
        filepath = os.path.join(directory, file_info["name"])

        result = read_file(filepath)
        if not result["success"]:
            print(f"Skipping {filepath}: {result['error']}")
            continue

        chunks = chunk_resume(result["content"])
        metadata = extract_metadata(chunks)

        processed.append({
            "filepath": filepath,
            "chunks": chunks,
            "metadata": metadata
        })

    return processed

def embed_texts(texts):
    """
    Takes a list of strings, returns a list of embedding vectors (one per string).
    """
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings


def store_resumes_in_chromadb(processed_resumes):
    """
    For every resume's chunks, embed each section and store it in ChromaDB
    with metadata so we can filter/trace results later.
    """
    ids = []
    documents = []
    metadatas = []

    for resume in processed_resumes:
        filepath = resume["filepath"]
        chunks = resume["chunks"]
        metadata = resume["metadata"]

        for section_name, section_text in chunks.items():
            if not section_text:
                continue  # skip empty sections

            chunk_id = f"{filepath}::{section_name}"
            ids.append(chunk_id)
            documents.append(section_text)
            metadatas.append({
                "filepath": filepath,
                "section": section_name,
                "name": metadata["name"],
                "skills": metadata["skills"],
                "years_experience": metadata["years_experience"],
                "education": metadata["education"],
            })

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Stored {len(documents)} chunks in ChromaDB.")

def test_query(query_text, n_results=3):
    """
    Quick sanity check: embed a query and see what comes back.
    """
    query_embedding = co.embed(
        texts=[query_text],
        model="embed-english-v3.0",
        input_type="search_query"   # note: different input_type than storing documents
    ).embeddings[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    for i in range(len(results["ids"][0])):
        print(f"\n--- Match {i+1} ---")
        print("ID:", results["ids"][0][i])
        print("Distance:", results["distances"][0][i])
        print("Section:", results["metadatas"][0][i]["section"])
        print("Name:", results["metadatas"][0][i]["name"])
        print("Text snippet:", results["documents"][0][i][:150])

if __name__ == "__main__":
    # all_resumes = load_and_process_resumes("resumes")
    # store_resumes_in_chromadb(all_resumes)
    print("Total items in ChromaDB:", collection.count())