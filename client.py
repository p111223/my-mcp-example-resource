import asyncio
import json
import sys
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 中的环境变量


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # 初始化百炼（DashScope）的 OpenAI 兼容客户端
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 工具服务器（如 weather.py）"""
        if not (server_script_path.endswith('.py') or server_script_path.endswith('.js')):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """使用百炼大模型 + MCP 工具处理用户查询"""
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": query}
        ]

        # 获取可用工具（转换为 OpenAI 格式）
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema  # 必须是合法 JSON Schema
                }
            }
            for tool in response.tools
        ]

        # 第一次调用模型
        response = await self.client.chat.completions.create(
            model="qwen-max",  # 可替换为 qwen-plus / qwen-turbo
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            max_tokens=1000
        )

        message = response.choices[0].message
        final_text_parts = []

        # 处理模型直接回复（无工具调用）
        if message.content:
            final_text_parts.append(message.content)

        # 如果有工具调用
        if message.tool_calls:
            # 将助手消息加入历史（含 tool_calls）
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # 逐个执行工具
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    final_text_parts.append(f"[Error parsing arguments for {func_name}: {e}]")
                    continue

                try:
                    tool_result = await self.session.call_tool(func_name, func_args)

                    # 安全提取文本内容
                    if hasattr(tool_result, 'content'):
                        content = tool_result.content
                        if isinstance(content, str):
                            result_content = content
                        elif isinstance(content, list):
                            # 假设每个 item 有 .text 属性（MCP TextContent）
                            texts = []
                            for item in content:
                                if hasattr(item, 'text'):
                                    texts.append(item.text)
                                else:
                                    texts.append(str(item))
                            result_content = "\n".join(texts)
                        else:
                            result_content = str(content)
                    else:
                        result_content = str(tool_result)


                except Exception as e:
                    result_content = f"[Tool execution error: {str(e)}]"

                # 添加工具返回结果到消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

                final_text_parts.append(f"[Called tool '{func_name}' with args {func_args}]")
                final_text_parts.append(result_content)

            # 再次调用模型生成最终回答
            final_response = await self.client.chat.completions.create(
                model="qwen-max",
                messages=messages,
                max_tokens=1000
            )
            final_message = final_response.choices[0].message
            if final_message.content:
                final_text_parts.append(final_message.content)

        return "\n".join(final_text_parts)

    async def chat_loop(self):
        """交互式聊天循环"""
        print("\nMCP Client Started (using Bailian/Qwen via DashScope)!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print("\nResponse:\n" + response)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
