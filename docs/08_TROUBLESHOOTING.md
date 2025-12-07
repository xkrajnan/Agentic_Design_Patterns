# Troubleshooting Guide

## Common Errors and Solutions

---

## 1. Jupyter Notebook Event Loop Issues

### Error
```
RuntimeError: This event loop is already running
```

### Cause
Jupyter notebooks already have a running event loop. Using `asyncio.run()` conflicts with it.

### Solution
```python
import nest_asyncio
nest_asyncio.apply()  # Must call BEFORE asyncio.run()

asyncio.run(main())  # Now works in notebook
```

**Alternative**: Direct await in notebooks
```python
# Instead of asyncio.run(main())
await main()  # Works directly in Jupyter
```

---

## 2. Template Variable Mismatch

### Error
```
UnresolvedVariableError: Variable 'my_results' not found in session state
```

### Cause
Mismatch between `output_key` and template variable name.

### Solution
```python
# ❌ WRONG - mismatch
agent1 = LlmAgent(..., output_key="my_result")
agent2 = LlmAgent(instruction="Use {my_results}")  # Note the 's'!

# ✅ CORRECT - exact match
agent1 = LlmAgent(..., output_key="my_result")
agent2 = LlmAgent(instruction="Use {my_result}")
```

**Debugging**:
```python
# Check session state contents
final_session = runner.get_session()
print("State keys:", final_session.state.keys())
# Verify your output_key is in the list
```

---

## 3. Missing output_key in ParallelAgent

### Error
```
AttributeError: ParallelAgent sub-agents must have output_key
```

### Cause
Parallel agents require `output_key` to store results independently.

### Solution
```python
# ❌ WRONG - no output_keys
ParallelAgent(sub_agents=[
    LlmAgent(name="A", ...),  # Missing!
    LlmAgent(name="B", ...)
])

# ✅ CORRECT - all have output_keys
ParallelAgent(sub_agents=[
    LlmAgent(name="A", ..., output_key="result_a"),
    LlmAgent(name="B", ..., output_key="result_b")
])
```

---

## 4. Agent Hierarchy Problems

### Error
```
AssertionError: greeter.parent_agent is None
```

### Cause
Agent not added to parent's `sub_agents` list.

### Solution
```python
# ❌ WRONG - parent-child relationship not established
greeter = LlmAgent(name="Greeter", ...)
coordinator = LlmAgent(name="Coordinator", ...)  # Missing sub_agents!

# ✅ CORRECT - establish relationship via sub_agents
greeter = LlmAgent(name="Greeter", ...)
coordinator = LlmAgent(
    name="Coordinator",
    sub_agents=[greeter]  # Now greeter.parent_agent == coordinator
)
```

---

## 5. API Configuration Issues

### Error (Google ADK)
```
AuthenticationError: Invalid API key
```

### Solution
```python
# Set environment variable
export GOOGLE_API_KEY="your-api-key"

# Or in Python
import os
os.environ["GOOGLE_API_KEY"] = "your-key"
```

### Error (LangChain + OpenAI)
```
EnvironmentError: OPENAI_API_KEY not found
```

### Solution
```python
# Use .env file
# Create .env with: OPENAI_API_KEY=your-key

from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
```

---

## 6. Code Block Parsing Failures

### Error
```
SyntaxError: invalid syntax (code includes markdown backticks)
```

### Cause
LLM returns code wrapped in markdown ```python ... ``` blocks.

### Solution
```python
def clean_code_block(code: str) -> str:
    """Remove markdown wrappers."""
    lines = code.strip().splitlines()
    
    # Remove leading ```python or ```
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    
    # Remove trailing ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    
    return "\\n".join(lines).strip()

# Usage
raw_code = llm.invoke(prompt).content
clean_code = clean_code_block(raw_code)  # Safe to execute
```

---

## 7. State Management Bugs

### Issue
Previous agent's result not available to next agent.

### Common Causes

**Cause 1**: Forgetting `output_key`
```python
# ❌ WRONG
agent1 = LlmAgent(...)  # No output_key!
agent2 = LlmAgent(instruction="Use {data}")  # {data} undefined!

# ✅ CORRECT
agent1 = LlmAgent(..., output_key="data")
agent2 = LlmAgent(instruction="Use {data}")
```

**Cause 2**: Sequential agents not waiting
```python
# ❌ WRONG - using ParallelAgent for dependent tasks
ParallelAgent(sub_agents=[extractor, transformer])
# transformer runs before extractor finishes!

# ✅ CORRECT - use SequentialAgent for dependencies
SequentialAgent(sub_agents=[extractor, transformer])
```

---

## 8. Event Processing Issues

### Issue
Missing code execution results or partial events.

### Solution
Always iterate through ALL parts:

```python
async for event in runner.run_async(...):
    if event.content and event.content.parts:
        for part in event.content.parts:  # Multiple parts!
            if part.executable_code:
                print(f"Code: {part.executable_code.code}")
            elif part.code_execution_result:
                print(f"Result: {part.code_execution_result.output}")
            elif part.text:
                print(f"Text: {part.text}")
```

**Don't assume**:
- ❌ `event.content.parts[0]` is the only part
- ❌ All events have `is_final_response()` = True
- ❌ Events come in a specific order

---

## 9. Tool Integration Errors

### Issue
Agent not using tool even though it's provided.

### Solutions

**1. Tool in wrong parameter**:
```python
# ❌ WRONG - code_executor in tools
LlmAgent(tools=[BuiltInCodeExecutor()])

# ✅ CORRECT - code_executor is separate parameter
LlmAgent(code_executor=BuiltInCodeExecutor())
```

**2. Instruction doesn't mention tool**:
```python
# ⚠️ LESS EFFECTIVE - no mention of tool
LlmAgent(
    instruction="Answer questions.",
    tools=[google_search]
)

# ✅ MORE EFFECTIVE - explicitly mentions tool
LlmAgent(
    instruction="Use the google_search tool to find information and answer questions.",
    tools=[google_search]
)
```

---

## 10. Vertex AI Search Issues

### Error
```
ValueError: DATASTORE_ID is None
```

### Solution
```python
import os

# Set environment variable
DATASTORE_ID = os.environ.get("DATASTORE_ID")

if not DATASTORE_ID:
    raise ValueError("Set DATASTORE_ID environment variable")

# Create agent
vsearch_agent = agents.VSearchAgent(
    datastore_id=DATASTORE_ID,  # Required!
    ...
)
```

---

## 11. Infinite Loop in LoopAgent

### Issue
LoopAgent never terminates, runs until max_iterations.

### Cause
Condition checker never yields `escalate=True`.

### Solution
```python
class ConditionChecker(BaseAgent):
    async def _run_async_impl(self, context):
        status = context.session.state.get("status", "pending")
        
        if status == "completed":
            # ✅ Signal termination
            yield Event(
                author=self.name,
                actions=EventActions(escalate=True)  # Critical!
            )
        else:
            yield Event(author=self.name, content="Continuing...")
```

**Always set max_iterations**:
```python
LoopAgent(
    max_iterations=10,  # Safety limit!
    sub_agents=[...]
)
```

---

## 12. Model Not Found Errors

### Error (Google ADK)
```
ModelNotFoundError: gemini-2.0-flash-exp not available
```

### Solution
Use stable model version:
```python
# Instead of experimental
model="gemini-2.0-flash"  # Stable version
```

### Error (OpenAI)
```
InvalidRequestError: Model 'gpt-4o' not found
```

### Solution
Check model access and use fallback:
```python
try:
    llm = ChatOpenAI(model="gpt-4o")
except:
    llm = ChatOpenAI(model="gpt-4-turbo")  # Fallback
```

---

## Debugging Checklist

When things don't work:

### 1. Check Environment
```python
# API keys set?
import os
print("GOOGLE_API_KEY:", "✅" if os.getenv("GOOGLE_API_KEY") else "❌")
print("OPENAI_API_KEY:", "✅" if os.getenv("OPENAI_API_KEY") else "❌")
```

### 2. Verify Agent Relationships
```python
# Parent-child correct?
print(f"Child's parent: {child_agent.parent_agent.name if child_agent.parent_agent else 'None'}")
```

### 3. Check Session State
```python
# State populated?
session = runner.get_session()
print("State keys:", list(session.state.keys()))
print("State values:", session.state)
```

### 4. Inspect Events
```python
# See all events
async for event in runner.run_async(...):
    print(f"Author: {event.author}")
    print(f"Content: {event.content}")
    print(f"Final?: {event.is_final_response()}")
```

### 5. Validate Template Variables
```python
# Do output_keys match template variables?
output_keys = ["renewable_energy_result", "ev_technology_result"]
instruction = "Use {renewable_energy_result} and {ev_technology_result}"

import re
template_vars = re.findall(r'{(\\w+)}', instruction)
print("Template vars:", template_vars)
print("Match?:", set(template_vars) == set(output_keys))
```

---

## Quick Fixes Reference

| Problem | Quick Fix |
|---------|-----------|
| Event loop error | `nest_asyncio.apply()` |
| Variable not found | Match `output_key` with `{template}` |
| Missing output_key | Add to ParallelAgent sub-agents |
| No parent_agent | Add to parent's `sub_agents=[]` |
| API auth error | Set environment variable |
| Code has backticks | Use `clean_code_block()` |
| State not shared | Use `output_key` + template variables |
| Tool not used | Mention in `instruction` |
| Infinite loop | Yield `EventActions(escalate=True)` |
| Model not found | Use stable version or fallback |

**Next**: Quick reference → [09_QUICK_REFERENCE.md](09_QUICK_REFERENCE.md)
