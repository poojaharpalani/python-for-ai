import asyncio
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client



#the server should run before this script - uv run
async def main():
     #connect server using sse 
    async with sse_client("http://127.0.0.1:8050/sse") as (read_stream, write_stream):
         
         #initialize the connection
          async with ClientSession(read_stream, write_stream) as session:
              await session.initialize()
              
              # list available tools
              tools_result = await session.list_tools()
              print(tools_result)
              print("Available tools:")
              for tool in tools_result.tools:
                   print(f"- {tool.name}: {tool.description}")
               
               
               #call calculator tool
              result = await session.call_tool("add", arguments= {"a": 2 , "b": 3})
              print(f"2+3 = {result.content[0].text}")
              
              
if __name__ == "__main__":
     asyncio.run(main())           
