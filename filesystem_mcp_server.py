import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None

mcp = FastMCP("filesystem")

# --- Configuration ---
ALLOWED_ROOT = Path(os.environ.get("FS_MCP_ROOT", "./resumes")).resolve()
MAX_FILE_SIZE_BYTES = int(os.environ.get("FS_MCP_MAX_FILE_SIZE", 5 * 1024 * 1024))  # 5 MB default
ALLOWED_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.environ.get("FS_MCP_ALLOWED_EXTENSIONS", ".txt,.pdf,.docx").split(",")
}

@mcp.resource("resumes://{filename}")
def get_resume(filename: str) -> str:
    """Expose a resume file as a readable MCP resource."""
    path = validate_path(filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resource not found: {filename}")
    with open(path, "r") as f:
        return f.read()


@mcp.resource("resumes://list")
def list_resume_resources() -> str:
    """List available resume resources as a newline-separated string."""
    files = [f for f in os.listdir(ALLOWED_ROOT) if os.path.isfile(os.path.join(ALLOWED_ROOT, f))]
    return "\n".join(sorted(files))

def validate_path(path: str) -> str:
    candidate = Path(path) if os.path.isabs(path) else ALLOWED_ROOT / path
    resolved = candidate.resolve()
    try:
        if os.path.commonpath([str(resolved), str(ALLOWED_ROOT)]) != str(ALLOWED_ROOT):
            raise ValueError(f"Access denied: '{path}' is outside the allowed directory")
    except ValueError as e:
        raise ValueError(f"Access denied: '{path}' is outside the allowed directory") from e
    return str(resolved)


def normalize_extension(ext):
    if ext is None:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower()


@mcp.tool()
def read_file(text_file: str) -> dict:
    """Read a .txt, .pdf, or .docx file and return its content and metadata.

    Args:
        text_file: Path to the file, relative to the server's allowed directory.
    """
    text_file = validate_path(text_file)

    if not os.path.isfile(text_file):
        return {"content": None, "success": False, "error": "File not found"}

    file_extension = os.path.splitext(text_file)[1].lower()
    file_size = os.path.getsize(text_file)
    modified_time = datetime.fromtimestamp(os.path.getmtime(text_file))

    if file_extension not in ALLOWED_EXTENSIONS:
        return {"content": None, "success": False, "error": f"Extension '{file_extension}' not allowed by server config"}

    if os.path.getsize(text_file) > MAX_FILE_SIZE_BYTES:
        return {"content": None, "success": False, "error": "File exceeds maximum allowed size"}

    if file_extension == ".txt":
        try:
            with open(text_file, "r") as f:
                content = f.read()
        except Exception as e:
            return {"content": None, "success": False, "error": str(e)}
    elif file_extension == ".pdf" and pypdf:
        try:
            with open(text_file, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
        except Exception as e:
            return {"content": None, "success": False, "error": str(e)}
    elif file_extension == ".docx" and docx:
        try:
            doc = docx.Document(text_file)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            return {"content": None, "success": False, "error": str(e)}
    else:
        return {"content": None, "success": False, "error": "Unsupported file extension"}

    metadata = {
        "extension": file_extension,
        "size_bytes": file_size,
        "modified_time": modified_time.isoformat(),
    }

    return {"content": content, "success": True, "metadata": metadata}


@mcp.tool()
def list_files(directory: str, extension: str | None = None) -> dict:
    """List files in a directory, optionally filtered by extension.

    Args:
        directory: Path to the directory, relative to the server's allowed directory.
        extension: Optional file extension filter, e.g. ".txt" or "pdf".
    """
    try:
        directory = validate_path(directory)
        if not os.path.isdir(directory):
            return {"files": None, "success": False, "error": "Directory not found"}

        target_ext = normalize_extension(extension)
        results = []

        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)

            if not os.path.isfile(full_path):
                continue

            file_ext = os.path.splitext(filename)[1].lower()
            if target_ext is not None and file_ext != target_ext:
                continue

            stat = os.stat(full_path)
            results.append({
                "name": filename,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return {"files": results, "success": True}

    except Exception as e:
        return {"files": None, "success": False, "error": str(e)}


@mcp.tool()
def write_file(filepath: str, content: str) -> dict:
    """Write text content to a file, creating parent folders if needed.

    Args:
        filepath: Path to write to, relative to the server's allowed directory.
        content: Text content to write.
    """
    try:
        filepath = validate_path(filepath)
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(filepath, "w") as f:
            f.write(content)

        return {"success": True, "filepath": filepath}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def search_in_file(filepath: str, keyword: str) -> dict:
    """Search for a keyword inside a file and return match positions with context.

    Args:
        filepath: Path to the file, relative to the server's allowed directory.
        keyword: The text to search for (case-insensitive).
    """
    file_result = read_file(filepath)

    if not file_result["success"]:
        return {"success": False, "error": file_result["error"]}

    content = file_result["content"]
    keyword_lower = keyword.lower()
    content_lower = content.lower()

    matches = []
    start_pos = 0

    while True:
        index = content_lower.find(keyword_lower, start_pos)
        if index == -1:
            break

        context_start = max(0, index - 30)
        context_end = min(len(content), index + len(keyword) + 30)
        context = content[context_start:context_end]

        matches.append({"position": index, "context": context})
        start_pos = index + len(keyword)

    return {"success": True, "matches": matches, "count": len(matches)}

_watch_state: dict[str, set[str]] = {}

@mcp.tool()
def watch_directory(directory: str = ".") -> dict:
    """Check a directory for new files since the last watch_directory call.

    Args:
        directory: Path to the directory to watch, relative to the server's allowed directory.
    """
    try:
        resolved = validate_path(directory)
        if not os.path.isdir(resolved):
            return {"success": False, "error": "Directory not found"}

        current_files = {f for f in os.listdir(resolved) if os.path.isfile(os.path.join(resolved, f))}
        previous_files = _watch_state.get(resolved, None)
        _watch_state[resolved] = current_files

        if previous_files is None:
            return {"success": True, "new_files": [], "message": "Baseline recorded", "total_files": len(current_files)}

        new_files = sorted(current_files - previous_files)
        return {"success": True, "new_files": new_files, "total_files": len(current_files)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
def batch_process(filenames: list[str]) -> dict:
    """Read multiple files and return per-file results, continuing past individual failures.

    Args:
        filenames: List of file paths, relative to the server's allowed directory.
    """
    results = []
    success_count = 0
    for name in filenames:
        result = read_file(name)
        results.append({"filename": name, **result})
        if result.get("success"):
            success_count += 1

    return {
        "success": True,
        "total": len(filenames),
        "succeeded": success_count,
        "failed": len(filenames) - success_count,
        "results": results,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")