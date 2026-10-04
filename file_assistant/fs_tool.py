import os
from datetime import datetime
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None 

def read_file(text_file):
    file_extension = os.path.splitext(text_file)[1].lower()
    file_size = os.path.getsize(text_file)
    modified_time = datetime.fromtimestamp(os.path.getmtime(text_file))

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
        "modified_time": modified_time
    }

    return {"content": content, "success": True, "metadata": metadata}  


def normalize_extension(ext):
    if ext is None:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower()

def list_files(directory, extension=None):
    try:
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
                "modified_time": datetime.fromtimestamp(stat.st_mtime)
            })

        return {"files": results, "success": True}

    except Exception as e:
        return {"files": None, "success": False, "error": str(e)}

def write_file(filepath, content):
    try:
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        with open(filepath, "w") as f:
            f.write(content)

        return {'success': True, 'filepath': filepath}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def search_in_file(filepath, keyword):
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

        matches.append({
            "position": index,
            "context": context
        })

        start_pos = index + len(keyword)

    return {
        "success": True,
        "matches": matches,
        "count": len(matches)
    }