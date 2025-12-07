# Tool Use Patterns: Extending Agent Capabilities

## Overview

This document covers three tool integration patterns from Chapter 5:

1. **Google Search Tool**: Web information retrieval
2. **Code Execution Tool**: Python code generation and execution  
3. **Vertex AI Search Tool**: Document/datastore queries

---

## 1. Google Search Tool Pattern

### Complete Runnable Example

```python
from google.adk.agents import Agent as ADKAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import nest_asyncio
import asyncio

nest_asyncio.apply()

# Configuration
APP_NAME = "Google_Search_agent"
USER_ID = "user1234"
SESSION_ID = "1234"

# Agent with google_search tool
root_agent = ADKAgent(
    name="basic_search_agent",
    model="gemini-2.0-flash-exp",
    description="Agent to answer questions using Google Search.",
    instruction="I can answer your questions by searching the internet. Just ask me anything!",
    tools=[google_search]  # Pre-built tool from google.adk.tools
)

# Agent Interaction
async def call_agent(query):
    # Session and Runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)

# Usage
asyncio.run(call_agent("what's the latest ai news?"))
```

**Key Points**:
- Use `from google.adk.tools import google_search`
- Add to `tools=[]` parameter
- Agent decides when to use tool based on query
- Synchronous execution with `runner.run()`

---

## 2. Code Execution Tool Pattern

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Configuration
APP_NAME = "calculator"
USER_ID = "user1234"
SESSION_ID = "session_code_exec_async"

# Agent with Code Executor
code_agent = LlmAgent(
    name="calculator_agent",
    model="gemini-2.0-flash",
    code_executor=BuiltInCodeExecutor(),  # Note: code_executor parameter, NOT tools
    instruction="""You are a calculator agent.
When given a mathematical expression, write and execute Python code to calculate the result.
Return only the final numerical result as plain text.""",
    description="Executes Python code to perform calculations."
)

# Async execution with event processing
async def call_agent_async(query):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(agent=code_agent, app_name=APP_NAME, session_service=session_service)
    content = types.Content(role='user', parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.executable_code:
                    print(f"Generated Code:\\n{part.executable_code.code}\\n")
                elif part.code_execution_result:
                    print(f"Result: {part.code_execution_result.output}")
                elif part.text:
                    print(f"Response: {part.text}")

# Usage
asyncio.run(call_agent_async("Calculate (5 + 7) * 3"))
```

**Event Part Types**:
- `part.executable_code.code`: Generated Python code
- `part.code_execution_result.output`: Execution output
- `part.code_execution_result.outcome`: Success/failure status
- `part.text`: Natural language response

---

## 3. Vertex AI Search Tool Pattern

### Complete Runnable Example

```python
from google.adk import agents
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import os

# Configuration
APP_NAME = "vsearch_app"
USER_ID = "user_123"
SESSION_ID = "session_456"
DATASTORE_ID = os.environ.get("DATASTORE_ID")  # Required!

# VSearchAgent - specialized for Vertex AI Search
vsearch_agent = agents.VSearchAgent(
    name="q2_strategy_vsearch_agent",
    description="Answers questions about Q2 strategy documents using Vertex AI Search.",
    model="gemini-2.0-flash-exp",
    datastore_id=DATASTORE_ID,  # Critical: Pass datastore ID
    model_parameters={"temperature": 0.0}  # Optional: Configure model
)

# Async streaming with grounding metadata
async def call_vsearch_agent_async(query: str):
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)

    try:
        runner = Runner(
            agent=vsearch_agent,
            app_name=APP_NAME,
            session_service=InMemorySessionService()
        )

        content = types.Content(role='user', parts=[types.Part(text=query)])

        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content
        ):
            # Token-by-token streaming
            if hasattr(event, 'content_part_delta') and event.content_part_delta:
                print(event.content_part_delta.text, end="", flush=True)

            # Process final response
            if event.is_final_response():
                print()  # Newline
                if event.grounding_metadata:
                    num_sources = len(event.grounding_metadata.grounding_attributions)
                    print(f"  (Sources: {num_sources})")
                print("-" * 30)

    except Exception as e:
        print(f"\\nError: {e}")

# Usage
if DATASTORE_ID:
    asyncio.run(call_vsearch_agent_async("Summarize Q2 strategy"))
else:
    print("Error: DATASTORE_ID environment variable not set")
```

**Key Features**:
- Uses `agents.VSearchAgent` (specialized subclass)
- Requires `datastore_id` parameter
- Provides `event.grounding_metadata` for source attribution
- Supports token streaming via `content_part_delta`

---

## Decision Matrix

| Need | Pattern | Tool/Parameter |
|------|---------|---------------|
| Web search | Google Search | `tools=[google_search]` |
| Generate & execute code | Code Execution | `code_executor=BuiltInCodeExecutor()` |
| Search documents/datastores | Vertex AI Search | `VSearchAgent` + `datastore_id` |

**Next**: See multi-agent collaboration → [05_MULTIAGENT_COLLABORATION.md](05_MULTIAGENT_COLLABORATION.md)
