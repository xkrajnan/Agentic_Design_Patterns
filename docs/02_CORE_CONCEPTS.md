# Core Concepts: Foundation for Agent

ic Design Patterns

## Overview

This document covers the fundamental concepts that underpin all agentic design patterns in Google ADK. Understanding these concepts is essential for implementing any multi-agent system.

## Core Concepts Covered

1. **Session State Management** - How agents share data
2. **Parent-Child Agent Relationships** - Agent hierarchies
3. **Event Handling and AsyncGenerators** - Communication mechanism
4. **Output Keys and Template Variables** - Data flow between agents
5. **Tool Integration** - Extending agent capabilities

---

## 1. Session State Management

### What is Session State?

**Session state** is a shared dictionary that persists across agent executions within a session. It enables agents to:
- Store intermediate results
- Share data with other agents
- Maintain context across multiple interactions

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

# Enable async in Jupyter notebooks
nest_asyncio.apply()

# Configuration
APP_NAME = \"state_demo\"
USER_ID = \"user123\"
SESSION_ID = \"session456\"

# Agent 1: Stores data in session state
agent1 = LlmAgent(
    name=\"DataCollector\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Extract the key information from the user's message and summarize it.\",
    output_key=\"collected_data\"  # Stores result in session.state[\"collected_data\"]
)

# Agent 2: Accesses data from session state
agent2 = LlmAgent(
    name=\"DataProcessor\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"\"\"Analyze the collected data and provide insights.

Data to analyze: {collected_data}

Provide 3 key insights.\"\"\"  # {collected_data} replaced with session.state[\"collected_data\"]
)

# Create sequential pipeline
pipeline = SequentialAgent(
    name=\"Pipeline\",
    sub_agents=[agent1, agent2]
)

# Execute
async def run_state_demo():
    # Create session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Create runner
    runner = Runner(
        agent=pipeline,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Prepare message
    content = types.Content(
        role='user',
        parts=[types.Part(text=\"I'm interested in learning about renewable energy, specifically solar and wind power.\")]
    )

    # Execute and print results
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.is_final_response():
            print(\"Final Response:\", event.content.parts[0].text)

# Run
asyncio.run(run_state_demo())
```

### Key Mechanisms

#### 1. Writing to State (via output_key)

```python
agent = LlmAgent(
    name=\"Writer\",
    output_key=\"my_result\"  # Writes agent output to session.state[\"my_result\"]
)
```

#### 2. Reading from State (via template variables)

```python
agent = LlmAgent(
    name=\"Reader\",
    instruction=\"Process this data: {my_result}\"  # {my_result} → session.state[\"my_result\"]
)
```

#### 3. Direct State Access (in custom agents)

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext

class CustomAgent(BaseAgent):
    async def _run_async_impl(self, context: InvocationContext):
        # Read from state
        previous_data = context.session.state.get(\"some_key\", \"default_value\")

        # Write to state
        context.session.state[\"new_key\"] = \"new_value\"

        # Yield result
        yield Event(author=self.name, content=\"Done\")
```

### Session Service Types

#### InMemorySessionService (Non-Persistent)

```python
session_service = InMemorySessionService()
# State lost when program ends
# Good for: Development, testing, single-run scripts
```

#### Persistent Session Services (Production)

```python
# Google ADK supports persistent session services
# State persists across runs
# Good for: Production, multi-user systems, chatbots
```

---

## 2. Parent-Child Agent Relationships

### What Are Parent-Child Relationships?

**Parent-child relationships** define agent hierarchies where:
- **Parent agents** delegate work to **child agents** (sub-agents)
- Children are specialized for specific tasks
- Parent coordinates and orchestrates children

### Hierarchy Structure

```
ParentAgent
├── ChildAgent1 (sub_agent)
├── ChildAgent2 (sub_agent)
└── ChildAgent3 (sub_agent)
```

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Child Agent 1: Specialized for greetings
greeter = LlmAgent(
    name=\"Greeter\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"You are a friendly greeter. Greet the user warmly.\"
)

# Child Agent 2: Specialized for farewells
farewell_agent = LlmAgent(
    name=\"FarewellExpert\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"You are a farewell specialist. Say goodbye politely.\"
)

# Parent Agent: Coordinator
coordinator = LlmAgent(
    name=\"Coordinator\",
    model=\"gemini-2.0-flash-exp\",
    description=\"A coordinator that delegates to specialized agents.\",
    instruction=\"\"\"When asked to greet someone, delegate to the Greeter.
When asked to say goodbye, delegate to the FarewellExpert.
Choose the appropriate agent based on the user's request.\"\"\",
    sub_agents=[greeter, farewell_agent]  # Define children
)

# Verify relationships
print(f\"Greeter's parent: {greeter.parent_agent.name if greeter.parent_agent else 'None'}\")
print(f\"FarewellExpert's parent: {farewell_agent.parent_agent.name if farewell_agent.parent_agent else 'None'}\")
# Output:
# Greeter's parent: Coordinator
# FarewellExpert's parent: Coordinator
```

### Relationship Properties

#### Automatic Binding

```python
# Framework automatically sets parent-child relationships
coordinator = LlmAgent(
    name=\"Parent\",
    sub_agents=[child1, child2]
)

# After creation:
assert child1.parent_agent == coordinator  # True
assert child2.parent_agent == coordinator  # True
```

#### Delegation Mechanism

1. **Explicit Delegation**: Parent's instruction tells it when to delegate
2. **Automatic Routing**: Framework handles actual delegation
3. **Result Return**: Child's output returned to parent

### Use Cases

| Pattern | Parent Role | Children Role |
|---------|-------------|---------------|
| **Coordinator** | Intelligent router | Specialized executors |
| **ParallelAgent** | Synchronization | Independent workers |
| **SequentialAgent** | Pipeline orchestrator | Sequential processors |
| **LoopAgent** | Iteration controller | Loop body executors |

---

## 3. Event Handling and AsyncGenerators

### What Are Events?

**Events** are the primary communication mechanism in Google ADK. Agents yield events to:
- Communicate results
- Stream partial outputs
- Signal state changes
- Control execution flow

### Event Types

```python
from google.adk.events import Event, EventActions

# 1. Content Event
yield Event(
    author=self.name,
    content=\"This is the result\"
)

# 2. Action Event (e.g., escalate to terminate loop)
yield Event(
    author=self.name,
    actions=EventActions(escalate=True)
)

# 3. Streaming Event (partial content)
yield Event(
    author=self.name,
    content_part_delta=\"token\"  # Streamed token-by-token
)
```

### AsyncGenerator Pattern

**AsyncGenerator** is Python's mechanism for async iteration. Custom agents must implement:

```python
from typing import AsyncGenerator
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

class CustomAgent(BaseAgent):
    name: str = \"CustomAgent\"
    description: str = \"Example custom agent\"

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        \"\"\"
        Async generator that yields Event objects.

        Args:
            context: Provides access to session, parent_agent, etc.

        Yields:
            Event objects for communication
        \"\"\"
        # Perform work
        result = await self._do_work(context)

        # Yield event
        yield Event(author=self.name, content=str(result))

    async def _do_work(self, context):
        # Your custom logic here
        return \"Work completed\"
```

### Complete Runnable Example: Custom Agent with Events

```python
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from typing import AsyncGenerator
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Custom agent that yields multiple events
class DataValidator(BaseAgent):
    name: str = \"DataValidator\"
    description: str = \"Validates data and yields progress events\"

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Get input from session state
        data = context.session.state.get(\"input_data\", \"\")

        # Yield progress event
        yield Event(author=self.name, content=\"Starting validation...\")

        # Perform validation
        is_valid = len(data) > 0 and \"test\" in data.lower()

        # Yield result event
        if is_valid:
            yield Event(author=self.name, content=\"✅ Data is valid\")
        else:
            yield Event(author=self.name, content=\"❌ Data is invalid\")

        # Store result in state
        context.session.state[\"validation_result\"] = \"valid\" if is_valid else \"invalid\"

# LLM agent that uses validation result
processor = LlmAgent(
    name=\"Processor\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"\"\"The data validation result is: {validation_result}

If valid, process the data: {input_data}
If invalid, explain why it failed validation.\"\"\"
)

# Create pipeline
pipeline = SequentialAgent(
    name=\"ValidationPipeline\",
    sub_agents=[
        DataValidator(),
        processor
    ]
)

# Execute
async def run_event_demo():
    APP_NAME = \"event_demo\"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=\"user1\",
        session_id=\"session1\"
    )

    # Pre-populate state with input
    session.state[\"input_data\"] = \"This is a test dataset\"

    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=\"Validate and process the data\")])

    print(\"Events stream:\")
    async for event in runner.run_async(user_id=\"user1\", session_id=\"session1\", new_message=content):
        print(f\"  [{event.author}]: {event.content.parts[0].text if event.content and event.content.parts else 'No content'}\")

asyncio.run(run_event_demo())
```

### Event Processing Patterns

#### 1. Collecting All Events

```python
events_list = []
async for event in runner.run_async(...):
    events_list.append(event)

# Process after completion
final_event = events_list[-1]
```

#### 2. Streaming (Real-time Processing)

```python
async for event in runner.run_async(...):
    if event.content_part_delta:
        print(event.content_part_delta.text, end=\"\", flush=True)
```

#### 3. Final Response Only

```python
async for event in runner.run_async(...):
    if event.is_final_response():
        print(event.content.parts[0].text)
        break  # Only care about final result
```

---

## 4. Output Keys and Template Variables

### What Are Output Keys?

**Output keys** are string identifiers that specify where an agent's result should be stored in session state.

### What Are Template Variables?

**Template variables** are placeholders in agent instructions (format: `{variable_name}`) that get replaced with values from session state.

### The Connection

```
Agent A (output_key=\"result_a\")
    ↓ (stores in session.state[\"result_a\"])
Session State: {\"result_a\": \"Value from A\"}
    ↓ (replaces {result_a} in instruction)
Agent B (instruction=\"Use {result_a}\")
```

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Parallel data gatherers (each stores in different output_key)
weather_agent = LlmAgent(
    name=\"WeatherGatherer\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Provide a brief weather summary for Tokyo.\",
    output_key=\"weather_data\"  # Stores in session.state[\"weather_data\"]
)

news_agent = LlmAgent(
    name=\"NewsGatherer\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Provide a brief news headline about technology.\",
    output_key=\"news_data\"  # Stores in session.state[\"news_data\"]
)

# Parallel execution
parallel_gatherer = ParallelAgent(
    name=\"DataGatherer\",
    sub_agents=[weather_agent, news_agent]
)

# Synthesis agent (uses template variables)
synthesizer = LlmAgent(
    name=\"Synthesizer\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"\"\"Create a brief daily summary combining this information:

Weather: {weather_data}

News: {news_data}

Format as a 2-sentence daily briefing.\"\"\"
    # {weather_data} and {news_data} are replaced with actual values from session state
)

# Sequential pipeline: gather in parallel, then synthesize
pipeline = SequentialAgent(
    name=\"DailyBriefingPipeline\",
    sub_agents=[parallel_gatherer, synthesizer]
)

# Execute
async def run_template_demo():
    APP_NAME = \"template_demo\"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=\"user1\",
        session_id=\"session1\"
    )

    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    content = types.Content(role='user', parts=[types.Part(text=\"Generate daily briefing\")])

    async for event in runner.run_async(user_id=\"user1\", session_id=\"session1\", new_message=content):
        if event.is_final_response():
            print(\"Daily Briefing:\")
            print(event.content.parts[0].text)

asyncio.run(run_template_demo())
```

### Important Rules

#### 1. Exact Name Matching

```python
# ✅ CORRECT - names match
agent.output_key = \"my_result\"
other_agent.instruction = \"Use {my_result}\"

# ❌ WRONG - mismatch causes error
agent.output_key = \"my_result\"
other_agent.instruction = \"Use {my_results}\"  # Note the 's'
```

#### 2. Output Keys Required for Parallel Agents

```python
# ✅ CORRECT - parallel agents must have output_keys
ParallelAgent(sub_agents=[
    LlmAgent(..., output_key=\"result1\"),
    LlmAgent(..., output_key=\"result2\"),
])

# ❌ WRONG - missing output_keys in parallel execution
ParallelAgent(sub_agents=[
    LlmAgent(...),  # No output_key!
    LlmAgent(...),
])
```

#### 3. Template Variables Must Exist

```python
# If {some_var} is in instruction, session.state[\"some_var\"] must exist
# Otherwise: UnresolvedVariable error
```

---

## 5. Tool Integration

### What Are Tools?

**Tools** extend agent capabilities by providing access to:
- External APIs (Google Search)
- Code execution environments
- Databases and datastores
- Custom functions

### Tool Types

1. **Pre-built Tools**: Provided by Google ADK
   - `google_search`: Web search
   - `BuiltInCodeExecutor`: Python code execution

2. **Custom Tools**: User-defined functions

3. **Agent Tools**: Agents wrapped as tools (AgentTool pattern)

### Pre-Built Tool Example: Google Search

```python
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Agent with google_search tool
search_agent = LlmAgent(
    name=\"SearchAgent\",
    model=\"gemini-2.0-flash-exp\",
    description=\"Agent that can search the web\",
    instruction=\"Use the google_search tool to find information and answer questions.\",
    tools=[google_search]  # Tool added here
)

async def run_search_demo():
    APP_NAME = \"search_demo\"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=\"user1\",
        session_id=\"session1\"
    )

    runner = Runner(agent=search_agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(
        role='user',
        parts=[types.Part(text=\"What are the latest developments in quantum computing?\")]
    )

    async for event in runner.run_async(user_id=\"user1\", session_id=\"session1\", new_message=content):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(run_search_demo())
```

### Pre-Built Tool Example: Code Execution

```python
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Agent with code execution capability
code_agent = LlmAgent(
    name=\"CodeExecutor\",
    model=\"gemini-2.0-flash-exp\",
    code_executor=BuiltInCodeExecutor(),  # Note: code_executor parameter, not tools
    instruction=\"\"\"You can write and execute Python code to solve problems.
When given a calculation, write Python code and execute it to get the result.\"\"\",
    description=\"Executes Python code\"
)

async def run_code_demo():
    APP_NAME = \"code_demo\"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=\"user1\",
        session_id=\"session1\"
    )

    runner = Runner(agent=code_agent, app_name=APP_NAME, session_service=session_service)

    content = types.Content(
        role='user',
        parts=[types.Part(text=\"Calculate the factorial of 10\")]
    )

    async for event in runner.run_async(user_id=\"user1\", session_id=\"session1\", new_message=content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.executable_code:
                    print(f\"Generated Code:\\n{part.executable_code.code}\\n\")
                elif part.code_execution_result:
                    print(f\"Execution Result: {part.code_execution_result.output}\")
                elif part.text:
                    print(f\"Response: {part.text}\")

asyncio.run(run_code_demo())
```

### Custom Tool Example

```python
# Define custom function
def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    \"\"\"
    Calculate tip and total bill.

    Args:
        bill_amount: Original bill amount in dollars
        tip_percentage: Tip percentage (default 15%)

    Returns:
        Dictionary with tip and total amounts
    \"\"\"
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {
        \"tip\": round(tip, 2),
        \"total\": round(total, 2)
    }

# Create agent with custom tool
tip_agent = LlmAgent(
    name=\"TipCalculator\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Use the calculate_tip function to help users calculate tips.\",
    tools=[calculate_tip]  # Add custom function as tool
)
```

### Tool vs Code Executor

| Aspect | Tools (tools parameter) | Code Executor (code_executor parameter) |
|--------|------------------------|----------------------------------------|
| **Purpose** | Call external functions/APIs | Generate & execute Python code |
| **Parameter** | `tools=[...]` | `code_executor=BuiltInCodeExecutor()` |
| **Examples** | google_search, custom functions | Built-in Python execution |
| **Control** | LLM decides when to call | LLM writes code, framework executes |
| **Sandboxing** | N/A | Yes, isolated execution environment |

---

## Summary: How Concepts Connect

```
Session State (persistent data store)
    ↕
Output Keys (write to state) ← Agent A → Template Variables (read from state)
    ↕
Events (communication) ← AsyncGenerators (yield mechanism)
    ↕
Parent-Child Relationships (hierarchy) ← sub_agents (composition)
    ↕
Tools (extend capabilities) ← tools parameter or code_executor
```

### Complete Integration Example

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Everything together:
# - Tools (google_search)
# - Output keys
# - Template variables
# - Parent-child relationships (ParallelAgent, SequentialAgent)
# - Session state
# - Event handling

# Parallel research (uses tools + output_keys)
researcher1 = LlmAgent(
    name=\"TechResearcher\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Research latest technology trends using google_search\",
    tools=[google_search],
    output_key=\"tech_trends\"
)

researcher2 = LlmAgent(
    name=\"BusinessResearcher\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Research business news using google_search\",
    tools=[google_search],
    output_key=\"business_news\"
)

parallel_research = ParallelAgent(
    name=\"ParallelResearch\",
    sub_agents=[researcher1, researcher2]  # Parent-child relationship
)

# Synthesis (uses template variables)
synthesizer = LlmAgent(
    name=\"Synthesizer\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"\"\"Combine these research findings into a brief report:

Technology Trends: {tech_trends}

Business News: {business_news}

Create a 3-sentence executive summary.\"\"\"
)

# Pipeline (sequential composition)
pipeline = SequentialAgent(
    name=\"ResearchPipeline\",
    sub_agents=[parallel_research, synthesizer]
)

# Execute with full event handling
async def run_complete_demo():
    APP_NAME = \"complete_demo\"
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=\"user1\",
        session_id=\"session1\"
    )

    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    content = types.Content(
        role='user',
        parts=[types.Part(text=\"Create a research report on current trends\")]
    )

    print(\"=== Event Stream ===\")
    async for event in runner.run_async(user_id=\"user1\", session_id=\"session1\", new_message=content):
        if event.author:
            print(f\"[{event.author}] Event received\")
        if event.is_final_response():
            print(\"\\n=== Final Report ===\")
            print(event.content.parts[0].text)

asyncio.run(run_complete_demo())
```

---

**Next Steps**:
- See specific patterns → [03-06] Pattern files
- Understand decision frameworks → [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)
- Troubleshoot issues → [08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)
