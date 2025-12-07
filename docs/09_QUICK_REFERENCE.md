# Quick Reference: Cheatsheet

## Import Statements

### Google ADK Imports

```python
# Core agents
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.agents import Agent as ADKAgent  # Alias

# Specialized agents
from google.adk import agents  # For VSearchAgent

# Runners and sessions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# Tools
from google.adk.tools import google_search, agent_tool
from google.adk.code_executors import BuiltInCodeExecutor

# Events and context
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

# Message types
from google.genai import types

# Async utilities
import asyncio
import nest_asyncio
from typing import AsyncGenerator
```

### LangChain + OpenAI Imports

```python
# LangChain
from langchain_openai import ChatOpenAI

# Environment
from dotenv import load_dotenv, find_dotenv
import os

# Utilities
import re
import random
from pathlib import Path
```

---

## Function Signatures

### Google ADK Agent Creation

```python
# LlmAgent
LlmAgent(
    name: str,                         # Required: agent identifier
    model: str,                        # Required: "gemini-2.0-flash-exp"
    instruction: str = "",             # System prompt
    description: str = "",             # Agent description
    tools: list[Tool] = [],            # Tools available
    code_executor: CodeExecutor = None,  # For code execution
    sub_agents: list[BaseAgent] = [],  # Child agents
    output_key: str = None,            # Where to store result
    model_parameters: dict = None      # Model config
)

# ParallelAgent
ParallelAgent(
    name: str,
    sub_agents: list[BaseAgent],       # Agents to run concurrently
    description: str = ""
)

# SequentialAgent
SequentialAgent(
    name: str,
    sub_agents: list[BaseAgent],       # Agents to run in order
    description: str = ""
)

# LoopAgent
LoopAgent(
    name: str,
    max_iterations: int,               # Safety limit
    sub_agents: list[BaseAgent],
    description: str = ""
)

# VSearchAgent
agents.VSearchAgent(
    name: str,
    model: str,
    description: str,
    datastore_id: str,                 # Required: Vertex AI datastore
    model_parameters: dict = None
)
```

### LangChain ChatOpenAI

```python
ChatOpenAI(
    model: str = "gpt-4o",             # Model ID
    temperature: float = 0.7,          # 0=deterministic, 1=creative
    openai_api_key: str = None,        # API key
    max_tokens: int = None,            # Token limit
    timeout: float = None              # Request timeout
)
```

### Runner Execution

```python
# Create runner
runner = Runner(
    agent: BaseAgent,
    app_name: str,
    session_service: SessionService
)

# Synchronous execution
events = runner.run(
    user_id: str,
    session_id: str,
    new_message: Content
) -> Iterator[Event]

# Asynchronous execution
async for event in runner.run_async(
    user_id: str,
    session_id: str,
    new_message: Content
) -> AsyncIterator[Event]:
    # Process events
```

---

## Configuration Templates

### Basic Google ADK Setup

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import nest_asyncio

nest_asyncio.apply()

# Config
APP_NAME = "my_app"
USER_ID = "user123"
SESSION_ID = "session456"
MODEL = "gemini-2.0-flash-exp"

# Agent
agent = LlmAgent(
    name="MyAgent",
    model=MODEL,
    instruction="Your instruction here"
)

# Execute
async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text="Your query")])
    
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(run())
```

### Basic LangChain Setup

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)

response = llm.invoke("Your prompt here")
print(response.content)
```

---

## Common Patterns

### 1. Parallel Data Gathering

```python
parallel = ParallelAgent(sub_agents=[
    LlmAgent(name="A", ..., output_key="result_a"),
    LlmAgent(name="B", ..., output_key="result_b"),
    LlmAgent(name="C", ..., output_key="result_c")
])
```

### 2. Sequential Pipeline

```python
pipeline = SequentialAgent(sub_agents=[
    LlmAgent(name="Extract", ..., output_key="extracted"),
    LlmAgent(name="Transform", instruction="Process {extracted}", output_key="transformed"),
    LlmAgent(name="Load", instruction="Store {transformed}")
])
```

### 3. Hybrid (Parallel → Sequential)

```python
hybrid = SequentialAgent(sub_agents=[
    ParallelAgent(sub_agents=[...]),  # Gather in parallel
    LlmAgent(name="Synthesizer", ...)  # Process sequentially
])
```

### 4. Tool Use

```python
from google.adk.tools import google_search

agent = LlmAgent(
    name="SearchAgent",
    model="gemini-2.0-flash-exp",
    instruction="Use google_search to find information",
    tools=[google_search]
)
```

### 5. Code Execution

```python
from google.adk.code_executors import BuiltInCodeExecutor

agent = LlmAgent(
    name="CodeAgent",
    model="gemini-2.0-flash",
    code_executor=BuiltInCodeExecutor(),
    instruction="Write and execute Python code"
)
```

### 6. Custom BaseAgent

```python
from google.adk.agents import BaseAgent
from typing import AsyncGenerator

class CustomAgent(BaseAgent):
    name: str = "CustomAgent"
    description: str = "Description"
    
    async def _run_async_impl(self, context) -> AsyncGenerator[Event, None]:
        # Your logic
        yield Event(author=self.name, content="Result")
```

### 7. Loop with Condition

```python
class ConditionChecker(BaseAgent):
    name: str = "Checker"
    
    async def _run_async_impl(self, context):
        done = context.session.state.get("done", False)
        if done:
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name, content="Continuing")

loop = LoopAgent(
    name="Loop",
    max_iterations=10,
    sub_agents=[process_agent, ConditionChecker()]
)
```

### 8. AgentTool Composition

```python
from google.adk.tools import agent_tool

sub_agent = LlmAgent(...)
tool = agent_tool.AgentTool(agent=sub_agent, description="Tool description")

parent = LlmAgent(tools=[tool])
```

### 9. LLM-as-Judge Pattern

```python
def goals_met(feedback: str, goals: list[str]) -> bool:
    llm = ChatOpenAI(model="gpt-4o")
    prompt = f"Goals: {goals}\\nFeedback: {feedback}\\nMet? True/False"
    response = llm.invoke(prompt).content.strip().lower()
    return response == "true"
```

---

## Pattern Comparison Table

| Pattern | File | Concurrency | Dependencies | Complexity |
|---------|------|-------------|--------------|------------|
| **Google Search** | 04 | No | None | Low |
| **Code Execution** | 04 | No | None | Low |
| **Vertex AI Search** | 04 | No | Datastore | Low |
| **ParallelAgent** | 03 | Yes | None | Medium |
| **SequentialAgent** | 03 | No | Strong | Low |
| **Coordinator** | 05 | No | None | High |
| **Loop** | 05 | No | Condition | Medium |
| **AgentTool** | 05 | Varies | None | Low |
| **Goal Setting** | 06 | No | None | Medium |

---

## File Location Index

```
docs/
├── 00_ROADMAP_INDEX.md             # Master index, pattern selector
├── 01_FRAMEWORKS_COMPARISON.md     # ADK vs LangChain, models
├── 02_CORE_CONCEPTS.md             # State, events, tools, relationships
├── 03_PARALLELIZATION_PATTERNS.md  # Parallel, Sequential, Hybrid
├── 04_TOOL_USE_PATTERNS.md         # Search, Code Exec, Vertex AI
├── 05_MULTIAGENT_COLLABORATION.md  # Coordinator, Loop, AgentTool
├── 06_GOAL_SETTING_ITERATION.md    # LLM-as-judge, refinement
├── 07_DECISION_FRAMEWORKS.md       # Flowcharts, matrices, trade-offs
├── 08_TROUBLESHOOTING.md           # Common errors and solutions
└── 09_QUICK_REFERENCE.md           # This file

notebooks/
├── Chapter 3_ Parallelization (Google ADK Code Example).ipynb
├── Chapter 5_ Tool Use (using Google Search).ipynb
├── Chapter 5_ Tool Use (Executing Code).ipynb
├── Chapter 5_ Tool Use (Vertex AI Search).ipynb
├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Coordinator).ipynb
├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Parallel).ipynb
├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Sequential).ipynb
├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Loop).ipynb
├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini AgentTooll).ipynb
└── Chapter 11_ Goal Setting and Monitoring (Goal_Setting_Iteration).ipynb
```

---

## Environment Variables

```bash
# Google ADK
export GOOGLE_API_KEY="your-google-api-key"
export DATASTORE_ID="your-vertex-ai-datastore-id"  # For Vertex AI Search

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Or use .env file:
# Create .env with:
# GOOGLE_API_KEY=...
# OPENAI_API_KEY=...
# DATASTORE_ID=...
```

---

## Common Utilities

### Message Construction

```python
from google.genai import types

content = types.Content(
    role='user',  # or 'model'
    parts=[types.Part(text="Your message text")]
)
```

### Event Checking

```python
# Check if final response
if event.is_final_response():
    result = event.content.parts[0].text

# Check event parts
for part in event.content.parts:
    if part.text:
        print(part.text)
    if part.executable_code:
        print(part.executable_code.code)
    if part.code_execution_result:
        print(part.code_execution_result.output)
```

### Session State Access

```python
# In custom agent
data = context.session.state.get("key", "default")
context.session.state["new_key"] = value

# After execution
session = runner.get_session()
print(session.state)
```

### Code Block Cleaning

```python
def clean_code_block(code: str) -> str:
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\\n".join(lines).strip()
```

---

## Model IDs

### Google (Gemini)
- `gemini-2.0-flash-exp` - Experimental, latest features
- `gemini-2.0-flash` - Stable version
- `gemini-2.0-flash-thinking-exp` - Reasoning-focused (if available)

### OpenAI
- `gpt-4o` - GPT-4 Omni (best quality)
- `gpt-4-turbo` - GPT-4 Turbo (faster, cheaper)
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Cheapest, good for simple tasks

---

## Performance Tips

1. **Use Parallel for independent tasks**
2. **Use low temperature (0.1-0.3) for deterministic outputs**
3. **Set max_iterations to prevent infinite loops**
4. **Use Gemini for high-volume, GPT-4o for quality-critical**
5. **Enable early termination with goals_met() or escalate**
6. **Batch similar operations**
7. **Cache expensive computations**

---

**For detailed documentation, see**: [00_ROADMAP_INDEX.md](00_ROADMAP_INDEX.md)
