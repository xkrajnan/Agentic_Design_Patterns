# Multi-Agent Collaboration: 5 Coordination Patterns

## Overview

Chapter 7 demonstrates 5 distinct multi-agent collaboration patterns:

1. **Coordinator**: Hierarchical delegation with intelligent routing
2. **Parallel**: Concurrent execution (covered in 03_PARALLELIZATION_PATTERNS.md)
3. **Sequential**: Linear pipelines (covered in 03_PARALLELIZATION_PATTERNS.md)
4. **Loop**: Iterative refinement with termination conditions
5. **AgentTool**: Agent-as-tool composition

---

## 1. Coordinator Pattern

### Concept

Parent agent intelligently delegates to specialized child agents based on task requirements.

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

# Custom Agent (extends BaseAgent)
class TaskExecutor(BaseAgent):
    """Specialized agent with custom, non-LLM behavior."""
    name: str = "TaskExecutor"
    description: str = "Executes a predefined task."

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """Custom implementation logic."""
        yield Event(author=self.name, content="Task finished successfully.")

# Child Agent 1: Greeter
greeter = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash-exp",
    instruction="You are a friendly greeter."
)

# Child Agent 2: Custom Task Executor
task_doer = TaskExecutor()

# Parent: Coordinator
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash-exp",
    description="A coordinator that can greet users and execute tasks.",
    instruction="When asked to greet, delegate to the Greeter. When asked to perform a task, delegate to the TaskExecutor.",
    sub_agents=[greeter, task_doer]
)

# Verify relationships
assert greeter.parent_agent == coordinator
assert task_doer.parent_agent == coordinator
print("✅ Agent hierarchy created successfully")
```

**Key Characteristics**:
- Parent makes intelligent delegation decisions
- Children are specialized for specific tasks
- Framework automatically establishes parent-child relationships

**When to Use**:
- Task routing based on content
- Specialized agents for different domains
- Need intelligent delegation logic

---

## 2. Loop Pattern

### Concept

Execute agents iteratively until a condition is met or max iterations reached.

### Complete Runnable Example

```python
import asyncio
from typing import AsyncGenerator
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

# Condition Checker (custom BaseAgent)
class ConditionChecker(BaseAgent):
    """Checks session state and signals loop termination."""
    name: str = "ConditionChecker"
    description: str = "Checks if process is complete."

    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Check state and yield escalate event if done."""
        status = context.session.state.get("status", "pending")
        is_done = (status == "completed")

        if is_done:
            # Signal loop termination
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            yield Event(author=self.name, content="Continuing loop...")

# Processing Agent
process_step = LlmAgent(
    name="ProcessingStep",
    model="gemini-2.0-flash-exp",
    instruction="Perform task. Set session state 'status' to 'completed' when done."
)

# Loop Orchestrator
poller = LoopAgent(
    name="StatusPoller",
    max_iterations=10,  # Safety limit
    sub_agents=[
        process_step,
        ConditionChecker()
    ]
)
```

**Event Actions**:
- `EventActions(escalate=True)`: Terminates loop early
- Loop continues until escalate signal OR max_iterations

**When to Use**:
- Iterative refinement tasks
- Polling until condition met
- Convergence scenarios

---

## 3. AgentTool Pattern

### Concept

Wrap an agent as a tool, allowing parent agents to invoke it like any other tool.

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from google.genai import types

# Core Function Tool
def generate_image(prompt: str) -> dict:
    """
    Generates image from text prompt.

    Args:
        prompt: Detailed description

    Returns:
        Dictionary with status and image_bytes
    """
    print(f"TOOL: Generating image for: '{prompt}'")
    mock_image_bytes = b"mock_image_data"
    return {
        "status": "success",
        "image_bytes": mock_image_bytes,
        "mime_type": "image/png"
    }

# Sub-Agent (specialized)
image_generator_agent = LlmAgent(
    name="ImageGen",
    model="gemini-2.0-flash",
    description="Generates image based on text prompt.",
    instruction="""Use the generate_image tool to create the image.
The user's request should be used as the 'prompt' argument.""",
    tools=[generate_image]
)

# Wrap Agent as Tool
image_tool = agent_tool.AgentTool(
    agent=image_generator_agent,
    description="Use this tool to generate an image. Input should be a prompt."
)

# Parent Agent (uses agent as tool)
artist_agent = LlmAgent(
    name="Artist",
    model="gemini-2.0-flash",
    instruction="""You are a creative artist.
First, invent a creative prompt for an image.
Then, use the ImageGen tool to generate the image.""",
    tools=[image_tool]  # Agent wrapped as tool
)
```

**Execution Flow**:
```
artist_agent receives task
    ↓
artist_agent calls ImageGen tool
    ↓
AgentTool invokes image_generator_agent
    ↓
image_generator_agent calls generate_image function
    ↓
Result returned through AgentTool to artist_agent
```

**When to Use**:
- Building modular, composable systems
- Agent reusability across different parents
- Mixing agents and functions as tools
- Multi-level agent hierarchies

---

## Pattern Comparison Matrix

| Pattern | Execution | Decision Making | Termination | Use Case |
|---------|-----------|----------------|-------------|----------|
| **Coordinator** | Sequential | Intelligent (LLM) | All children complete | Task routing |
| **Parallel** | Concurrent | None (all run) | All complete | Independent data gathering |
| **Sequential** | Sequential | None (ordered) | Last completes | Data pipelines |
| **Loop** | Iterative | Condition-based | Escalate or max iterations | Refinement loops |
| **AgentTool** | Depends on parent | Parent decides | Tool return | Modular composition |

---

## Complementarity Matrix

Patterns that work well together:

| Pattern 1 | Pattern 2 | Combination Benefit |
|-----------|-----------|-------------------|
| **Parallel** | **Sequential** | Gather concurrently, process sequentially |
| **Loop** | **Coordinator** | Iterative refinement with specialized agents |
| **AgentTool** | **Coordinator** | Modular agents coordinated intelligently |
| **Sequential** | **Loop** | Multi-step pipeline with iterative refinement |

---

## Orthogonality Analysis

**Independent Dimensions**:

1. **Execution Model**: Parallel, Sequential, Iterative
   - Can choose any based on task dependencies
   
2. **Communication Model**: State-based, Event-based, Tool-based
   - Orthogonal to execution model

3. **Decision Making**: Rule-based, LLM-based, Condition-based
   - Independent of execution and communication

4. **Tool Integration**: Pre-built, Custom, Agent-as-tool
   - Can be added to any pattern

**Example**: You can have:
- Sequential execution (dimension 1)
- With state-based communication (dimension 2)
- Using LLM-based decisions (dimension 3)
- With custom tools (dimension 4)

---

## Custom BaseAgent Template

```python
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

class CustomAgent(BaseAgent):
    """Template for custom agent implementation."""
    
    name: str = "CustomAgentName"
    description: str = "What this agent does"
    
    async def _run_async_impl(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Main execution logic.
        
        Args:
            context: Provides session, parent_agent, input
            
        Yields:
            Event objects for communication
        """
        try:
            # Access session state
            input_data = context.session.state.get("input", None)
            
            # Perform work
            result = await self._perform_task(input_data)
            
            # Yield result event
            yield Event(author=self.name, content=str(result))
            
        except Exception as e:
            yield Event(author=self.name, content=f"Error: {e}")
    
    async def _perform_task(self, data):
        """Helper method for actual work."""
        return f"Processed: {data}"
```

---

## Decision Framework

**Choose Coordinator when**:
- Parent needs to route tasks intelligently
- Children are highly specialized
- Task selection depends on content

**Choose Loop when**:
- Need iterative refinement
- Convergence condition is checkable
- May need multiple passes

**Choose AgentTool when**:
- Building modular systems
- Agents are reusable components
- Need multi-level nesting

**Next**: See goal setting → [06_GOAL_SETTING_ITERATION.md](06_GOAL_SETTING_ITERATION.md)
