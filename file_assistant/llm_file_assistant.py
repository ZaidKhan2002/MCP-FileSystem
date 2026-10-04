import os
from openai import OpenAI
import json

from file_assistant.fs_tool import read_file, list_files, write_file, search_in_file

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a resume file (PDF, TXT, or DOCX) and return its text content with metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory, optionally filtered by file extension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory to list files from"
                    },
                    "extension": {
                        "type": "string",
                        "description": "Optional file extension to filter by, e.g. '.pdf'"
                    }
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file, creating directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write to the file"
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "Search for a keyword inside a file's content and return matches with surrounding context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to search in"
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for (case-insensitive)"
                    }
                },
                "required": ["filepath", "keyword"]
            }
        }
    }
]

available_functions = {
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "search_in_file": search_in_file
}

def run_query(user_query):
    messages = [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        tools=tools
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_to_call = available_functions[function_name]

            result = function_to_call(**function_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

        second_response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages
        )

        print(second_response.choices[0].message.content)

    else:
        print(message.content)


run_query("List all files in the current directory")