"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from research_agent.prompts import (  # research_agent/prompts.py
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import tavily_search, think_tool  # research_agent/tools.py

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# 完整的研究工作流程指令，包含引用文献要求 这里提示词一行都不能少！！！
FULL_RESEARCH_INSTRUCTIONS = f"""{RESEARCH_WORKFLOW_INSTRUCTIONS}

## File Writing Requirements

**CRITICAL**: You MUST use the write_file tool to save files. Do NOT just output text.

1. **Save research request**: Use write_file("/research_request.md", content) to save the user's research question
2. **Save final report**: Use write_file("/final_report.md", content) to save your comprehensive report
**IMPORTANT**:
- Use absolute paths starting with "/" (e.g., "/research_request.md", "/final_report.md")
- The final report MUST include proper citations in [1], [2], [3] format
- The final report MUST end with a ### Sources section listing all numbered sources
- Each source should be formatted as: [1] Source Title: URL
- Do NOT just output text - you MUST use the write_file tool!

{SUBAGENT_DELEGATION_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
)}
"""

def make_backend(runtime):
    """创建复合存储后端."""
    base_dir = Path(__file__).parent
    research_dir = base_dir / "research_outputs"
    research_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Backend configured. Files will be saved to: {research_dir}")
    print(f"📁 Absolute path: {research_dir.absolute()}")
    
    # 使用 virtual_mode=True，这样绝对路径会解析到 root_dir 下
    fs_backend = FilesystemBackend(
        root_dir=str(research_dir.absolute()),
        virtual_mode=True  # 改为 True，这样 /research_request.md 会解析到 root_dir/research_request.md
    )
    
    return CompositeBackend(
        default=fs_backend,
        routes={"/memories/": StoreBackend(runtime)}
    )
# 创建模型（添加重试配置）
model = ChatOpenAI(
    base_url=os.getenv("DASHSCOPE_API_BASE"),
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    model="qwen-max-2025-01-25", # 选好模型很关键，差模型会fail
    timeout=60.0,
    max_retries=3,
)

# 构建research sub-agent字典
research_sub_agent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date) ,
    "tools": [tavily_search, think_tool],
    "model": model
}



# 创建持久化存储
store = InMemoryStore()

# Create the agent with official backends support
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],  # 主代理也拥有搜索工具，但会优先使用子代理
    backend=make_backend,
    system_prompt=FULL_RESEARCH_INSTRUCTIONS,
    subagents=[research_sub_agent],
    store=store,
    debug=True
)

async def run_with_retry(agent, messages, max_retries=3):
    """带重试机制的 agent 执行."""
    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempt {attempt + 1}/{max_retries}...")  # noqa: T201
            result = await agent.ainvoke({"messages": messages})
            return result
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")  # noqa: T201
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2)
    return None

if __name__ == "__main__":
    print("🚀 Starting research agent...")  # noqa: T201
    print(f"📝 Research topic: 'context engineering approaches used to build AI agents'")  # noqa: F541, T201
    
    research_output_dir = Path(__file__).parent / "research_outputs"
    research_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理之前的文件（因为 write_file 不覆盖已存在的文件）
    for file in research_output_dir.glob("*.md"):
        file.unlink()
        print(f"🧹 Cleaned up: {file.name}")  # noqa: T201
    
    try:
        # 运行 agent
        messages = [
            {
                "role": "user",
                "content": "research context engineering approaches used to build AI agents",
            }
        ]
        
        result = asyncio.run(run_with_retry(agent, messages))
        
        print("\n📋 Agent execution completed. Messages:")
        print("=" * 80)
        
        # 统计消息类型
        human_count = ai_count = tool_count = 0
        write_file_calls = []
        
        for msg in result["messages"]:
            if isinstance(msg, HumanMessage):
                human_count += 1
                print(f"\n👤 Human message {human_count}:")
                print(f"   {msg.content}")
            elif isinstance(msg, AIMessage):
                ai_count += 1
                print(f"\n🤖 AI message {ai_count}:")
                if hasattr(msg, 'content'):
                    content = msg.content
                    print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
                # 检查工具调用
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"   Tool calls: {len(msg.tool_calls)}")
                    for i, tc in enumerate(msg.tool_calls):
                        tool_name = tc['name']
                        print(f"   [{i+1}] Tool: {tool_name}")
                        if tool_name == 'write_file':
                            write_file_calls.append(tc)
                            print(f"       Arguments: {tc.get('args', {})}")
                        elif tool_name == 'task':
                            print(f"       ✅ Task tool called! Arguments: {tc.get('args', {})}")
                        else:
                            print(f"       Arguments: {tc.get('args', {})}")
                else:
                    print(f"   No tool calls")
            elif isinstance(msg, ToolMessage):
                tool_count += 1
                print(f"\n🔧 Tool message {tool_count}:")
                if hasattr(msg, 'name'):
                    tool_name = msg.name
                    print(f"   Tool: {tool_name}")
                    print(f"   Content: {msg.content[:200]}...")
                else:
                    print(f"   Content: {msg.content[:200]}...")
        
        print(f"\n📊 Summary:")
        print(f"   Human messages: {human_count}")
        print(f"   AI messages: {ai_count}")
        print(f"   Tool messages: {tool_count}")
        print(f"   write_file calls: {len(write_file_calls)}")
        
        # 检查是否生成了文件
        print("\n" + "=" * 80)
        print("📁 Checking for generated files:")
        
        research_request_path = research_output_dir / "research_request.md"
        final_report_path = research_output_dir / "final_report.md"
        
        if research_request_path.exists():
            print(f"✅ Generated research_request.md at: {research_request_path}")
            with open(research_request_path, encoding='utf-8') as f:
                content = f.read()
                print(f"   Content preview: {content[:150]}...")
        else:
            print("❌ research_request.md NOT generated")
            print("   Expected at: {research_request_path}")
        
        if final_report_path.exists():
            print(f"✅ Generated final_report.md at: {final_report_path}")
            with open(final_report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   Content preview: {content[:150]}...")
        else:
            print(f"❌ final_report.md NOT generated")
            print(f"   Expected at: {final_report_path}")
        
        # 列出 research_outputs 目录中的所有文件
        print(f"\n📁 All files in research_outputs directory ({research_output_dir}):")
        if research_output_dir.exists():
            files = list(research_output_dir.iterdir())
            if files:
                for file in files:
                    print(f"   - {file.name} ({file.stat().st_size} bytes)")
            else:
                print("   Directory is empty!")
        else:
            print("   Directory does not exist!")
            
        # 如果没有生成文件，提供诊断信息
        if not research_request_path.exists() or not final_report_path.exists():
            print("\n⚠️  DIAGNOSIS:")
            print("   Files were not generated. Possible reasons:")
            print("   1. Agent did not call write_file tool")
            print("   2. Tool call failed silently")
            print("   3. Backend configuration issue")
            print(f"   4. write_file calls detected: {len(write_file_calls)}")
            
    except Exception as e:
        print(f"❌ Error during agent execution: {e}")
        import traceback
        traceback.print_exc()