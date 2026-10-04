import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=["filesystem_mcp_server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            resources = await session.list_resources()
            print("Available resources:", [r.uri for r in resources.resources])

            result = await session.call_tool("list_files", {"directory": ".", "extension": None})
            print("list_files:", result.content)

            result = await session.call_tool("read_file", {"text_file": "jd_01.txt"})
            print("read_file:", result.content)

            result = await session.call_tool("batch_process", {"filenames": ["jd_01.txt", "missing.txt"]})
            print("batch_process:", result.content)

if __name__ == "__main__":
    asyncio.run(main())