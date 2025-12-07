# Framework Comparison: Google ADK vs LangChain/OpenAI

## Overview

This repository demonstrates two distinct approaches to building agentic systems:
- **Google Agent Development Kit (ADK)**: Structured multi-agent orchestration (Chapters 3, 5, 7)
- **LangChain + OpenAI**: Iterative refinement and prompt engineering (Chapter 11)

## Quick Comparison Table

| Aspect | Google ADK | LangChain + OpenAI |
|--------|------------|-------------------|
| **Primary Use Case** | Multi-agent orchestration & composition | Single-agent iterative refinement |
| **Models** | Gemini 2.0 Flash (Exp), Gemini 2.0 Flash | GPT-4o, GPT-4 Turbo |
| **Architecture** | Hierarchical agents (parent-child) | Monolithic function with internal loop |
| **State Management** | Session service (InMemorySessionService) | In-memory variables (previous_code, feedback) |
| **Execution** | Async/await with event streaming | Synchronous LLM invocations |
| **Concurrency** | Native (ParallelAgent, LoopAgent) | Sequential only |
| **Tool Integration** | Built-in (`google_search`, `BuiltInCodeExecutor`) | Function-based (manual integration) |
| **Composition** | Agent nesting via `sub_agents` | Not built-in |
| **Iteration Support** | LoopAgent with escalate signals | Manual for loops |
| **Code Complexity** | Higher (more classes, relationships) | Lower (simple functions) |
| **Best For** | Complex multi-step workflows | Quality-driven single tasks |

## Google ADK Deep Dive

### Philosophy & Design

**Core Principle**: Composition and orchestration of specialized agents

**Architecture**:
```
BaseAgent (abstract)
├── LlmAgent (LLM-powered agent)
│   ├── ParallelAgent (concurrent orchestrator)
│   ├── SequentialAgent (pipeline coordinator)
│   └── LoopAgent (iterative executor)
└── CustomAgent (user-defined, extends BaseAgent)
```

**Key Strengths**:
1. **Native multi-agent support**: Built-in parallelization and sequencing
2. **Session-based state**: Persistent state across agent executions
3. **Event streaming**: Real-time async event handling
4. **Tool ecosystem**: Pre-built tools (google_search, code execution, Vertex AI)
5. **Agent hierarchy**: Parent-child relationships for modular design

**When to Use Google ADK**:
- ✅ Multiple agents need to work together
- ✅ Concurrent execution required (parallel data gathering)
- ✅ Complex workflows with multiple steps
- ✅ Need event streaming for real-time feedback
- ✅ Using Google Cloud ecosystem (Vertex AI)

### Google ADK Code Pattern

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import asyncio

# Configuration
APP_NAME = \"my_app\"
USER_ID = \"user123\"
SESSION_ID = \"session456\"

# Create agents
researcher = LlmAgent(
    name=\"Researcher\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Research the topic using google_search tool\",
    tools=[google_search],
    output_key=\"research_data\"  # Store result in session state
)

analyzer = LlmAgent(
    name=\"Analyzer\",
    model=\"gemini-2.0-flash-exp\",
    instruction=\"Analyze the research: {research_data}\"  # Access via template
)

# Create pipeline
pipeline = SequentialAgent(
    name=\"Pipeline\",
    sub_agents=[researcher, analyzer]
)

# Execute
async def run():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(agent=pipeline, app_name=APP_NAME, session_service=session_service)

    content = types.Content(
        role='user',
        parts=[types.Part(text=\"Research AI trends\")]
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(run())
```

### Google ADK Execution Models

#### 1. Synchronous Execution
```python
events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
for event in events:
    if event.is_final_response():
        print(event.content.parts[0].text)
```
- Returns iterator of events
- Blocks until completion
- Simpler for straightforward tasks

#### 2. Asynchronous Execution
```python
async for event in runner.run_async(user_id, session_id, new_message):
    # Process events in real-time
    if event.content_part_delta:
        print(event.content_part_delta.text, end=\"\", flush=True)
```
- Streams events as they occur
- Non-blocking
- Better for concurrent operations

---

## LangChain + OpenAI Deep Dive

### Philosophy & Design

**Core Principle**: Iterative refinement through LLM-powered evaluation

**Architecture**:
```
Main Function (run_code_agent)
├── generate_prompt()          # Construct LLM prompt
├── llm.invoke()               # Generate/refine code
├── clean_code_block()         # Parse response
├── get_code_feedback()        # Critique code
├── goals_met()                # Binary decision (LLM-as-judge)
└── save_code_to_file()        # Persist result
```

**Key Strengths**:
1. **LLM-as-judge pattern**: Use LLM to evaluate its own output
2. **Iterative refinement**: Automatic quality improvement loop
3. **Flexibility**: Simple Python functions, easy to customize
4. **Prompt engineering**: Fine-grained control over LLM behavior
5. **Deterministic convergence**: Low temperature for consistency

**When to Use LangChain + OpenAI**:
- ✅ Single task requiring iterative quality improvement
- ✅ Clear evaluation criteria (goals)
- ✅ Prompt engineering is critical
- ✅ Need fine-grained control over LLM behavior
- ✅ Using OpenAI models (GPT-4o)

### LangChain + OpenAI Code Pattern

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv(\"OPENAI_API_KEY\")

# Initialize LLM
llm = ChatOpenAI(
    model=\"gpt-4o\",
    temperature=0.3,  # Low = deterministic
    openai_api_key=OPENAI_API_KEY
)

# Helper functions
def generate_prompt(use_case: str, goals: list[str], previous_code: str = \"\", feedback: str = \"\") -> str:
    prompt = f\"\"\"You are an AI coding agent. Write Python code for: {use_case}

Your goals are:
{chr(10).join(f\"- {g}\" for g in goals)}
\"\"\"
    if previous_code:
        prompt += f\"\\n\\nPreviously generated code:\\n{previous_code}\"
    if feedback:
        prompt += f\"\\n\\nFeedback on previous version:\\n{feedback}\"
    prompt += \"\\n\\nReturn only the revised Python code.\"
    return prompt

def get_code_feedback(code: str, goals: list[str]) -> str:
    feedback_prompt = f\"\"\"Review this code against these goals:
{chr(10).join(f\"- {g}\" for g in goals)}

Code:
{code}

Provide detailed critique.\"\"\"
    return llm.invoke(feedback_prompt)

def goals_met(feedback_text: str, goals: list[str]) -> bool:
    review_prompt = f\"\"\"Goals: {chr(10).join(f\"- {g}\" for g in goals)}

Feedback: {feedback_text}

Have the goals been met? Respond with only: True or False.\"\"\"
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == \"true\"

# Main loop
def run_code_agent(use_case: str, goals_input: str, max_iterations: int = 5) -> str:
    goals = [g.strip() for g in goals_input.split(\",\")]
    previous_code = \"\"
    feedback = \"\"

    for i in range(max_iterations):
        # Generate/refine code
        prompt = generate_prompt(use_case, goals, previous_code, feedback)
        code = llm.invoke(prompt).content.strip()

        # Get feedback
        feedback = get_code_feedback(code, goals).content.strip()

        # Check if goals met
        if goals_met(feedback, goals):
            print(\"✅ Goals met. Stopping.\")
            break

        previous_code = code

    return code

# Usage
code = run_code_agent(
    use_case=\"Write a function to calculate fibonacci numbers\",
    goals_input=\"Simple, Correct, Handles edge cases\"
)
print(code)
```

---

## Model Comparison: Gemini 2.0 vs GPT-4o

### Gemini 2.0 Flash (Google ADK)

**Variants**:
- `gemini-2.0-flash-exp`: Experimental, latest features
- `gemini-2.0-flash`: Stable version

**Characteristics**:
- **Speed**: Very fast inference
- **Cost**: Lower than GPT-4o
- **Strengths**: Tool use, multi-turn conversations
- **Integration**: Native with Google Cloud (Vertex AI, Google Search)
- **Context Window**: Large (up to 1M tokens in some variants)

**Best For**:
- High-throughput agent systems
- Integration with Google services
- Cost-sensitive applications
- Multi-agent orchestration

### GPT-4o (OpenAI)

**Model**: `gpt-4o` (GPT-4 Omni)

**Characteristics**:
- **Quality**: Highest reasoning capability
- **Cost**: Higher than Gemini
- **Strengths**: Complex reasoning, code generation, instruction following
- **Integration**: LangChain ecosystem, OpenAI tools
- **Context Window**: 128K tokens

**Best For**:
- Complex reasoning tasks
- High-quality code generation
- Iterative refinement with quality criteria
- Tasks requiring nuanced understanding

### When to Choose Which Model

| Scenario | Recommended Model | Reason |
|----------|------------------|--------|
| Multi-agent system with many API calls | Gemini 2.0 Flash | Lower cost, high throughput |
| Complex reasoning or code generation | GPT-4o | Superior reasoning quality |
| Integration with Google Cloud | Gemini 2.0 Flash | Native integration |
| Iterative refinement (few iterations) | GPT-4o | Better at following complex criteria |
| High-volume production system | Gemini 2.0 Flash | Cost-effective at scale |
| Prototype/research | Either | Use what you have access to |

---

## Architecture Philosophies

### Google ADK: Orchestration-First

**Metaphor**: Orchestra conductor coordinating specialized musicians

**Principles**:
1. **Separation of concerns**: Each agent has specific responsibility
2. **Composition over monolith**: Build complex systems from simple agents
3. **Explicit state management**: Session service handles persistence
4. **Event-driven**: Async events communicate results
5. **Declarative**: Define agent structure, framework handles execution

**Example**: Research paper writing system
```python
# Orchestration approach
pipeline = SequentialAgent(sub_agents=[
    ParallelAgent(sub_agents=[  # Gather data concurrently
        LlmAgent(name=\"MethodsResearcher\", ...),
        LlmAgent(name=\"ResultsResearcher\", ...),
        LlmAgent(name=\"RelatedWorkResearcher\", ...),
    ]),
    LlmAgent(name=\"Synthesizer\", ...),  # Combine research
    LlmAgent(name=\"Writer\", ...),        # Write paper
    LlmAgent(name=\"Reviewer\", ...)       # Review quality
])
```

### LangChain + OpenAI: Refinement-First

**Metaphor**: Craftsperson iteratively perfecting a single artifact

**Principles**:
1. **Iterative improvement**: Refine output until quality standards met
2. **LLM-as-judge**: Use LLM to evaluate its own work
3. **Prompt engineering**: Precise control via prompt design
4. **Procedural**: Explicit control flow (for loops, if statements)
5. **Convergence-focused**: Continue until goals satisfied

**Example**: High-quality code generation
```python
# Refinement approach
for iteration in range(max_iterations):
    code = llm.invoke(generate_prompt(use_case, goals, previous_code, feedback))
    feedback = llm.invoke(critique_prompt(code, goals))
    if goals_met(feedback):
        break
    previous_code = code
```

---

## Migration Strategies

### From Google ADK to LangChain/OpenAI

**When to Migrate**:
- Need GPT-4o's superior reasoning
- Simpler system (single agent sufficient)
- Want more control over prompts
- Iterative refinement is primary pattern

**Migration Pattern**:
```python
# Before (Google ADK)
agent = LlmAgent(
    name=\"Coder\",
    model=\"gemini-2.0-flash\",
    instruction=\"Generate code for {task}\"
)

# After (LangChain + OpenAI)
llm = ChatOpenAI(model=\"gpt-4o\", temperature=0.3)
response = llm.invoke(f\"Generate code for {task}\")
```

**State Migration**:
```python
# Before: Session-based state
context.session.state[\"result\"] = value

# After: Variables
result = value
```

### From LangChain/OpenAI to Google ADK

**When to Migrate**:
- Need multi-agent coordination
- Want parallel execution
- Require event streaming
- Building complex orchestration

**Migration Pattern**:
```python
# Before (LangChain)
for i in range(max_iterations):
    output = llm.invoke(prompt)
    feedback = llm.invoke(critique_prompt)
    if criteria_met(feedback):
        break

# After (Google ADK)
loop_agent = LoopAgent(
    max_iterations=max_iterations,
    sub_agents=[
        LlmAgent(name=\"Generator\", ...),
        ConditionChecker()  # Custom BaseAgent
    ]
)
```

---

## Framework Complementarity Analysis

### Using Both Frameworks Together

The two frameworks are **complementary**, not competing. You can use both in the same project:

#### Pattern 1: ADK for Orchestration, LangChain for Quality

```python
# Use ADK to orchestrate multiple tasks
parallel_tasks = ParallelAgent(sub_agents=[
    LlmAgent(name=\"Task1\", ...),  # Uses Gemini
    LlmAgent(name=\"Task2\", ...),  # Uses Gemini
])

# Use LangChain for quality-critical refinement
critical_output = run_iterative_refinement(
    task_description=\"...\",
    quality_goals=[\"...\"],
    llm=ChatOpenAI(model=\"gpt-4o\")
)
```

#### Pattern 2: ADK for Data Gathering, LangChain for Analysis

```python
# Step 1: Use ADK to gather data in parallel
data_gatherer = ParallelAgent(sub_agents=[...])
# Collect research_data from multiple sources

# Step 2: Use LangChain for deep analysis
analysis_llm = ChatOpenAI(model=\"gpt-4o\", temperature=0.1)
analysis = analysis_llm.invoke(f\"Analyze: {research_data}\")
```

#### Pattern 3: Hybrid Agent with Both Models

```python
# Gemini for initial draft (fast, cheap)
draft_agent = LlmAgent(
    name=\"Drafter\",
    model=\"gemini-2.0-flash\",
    instruction=\"Create initial draft\"
)

# GPT-4o for refinement (quality)
refiner_llm = ChatOpenAI(model=\"gpt-4o\")
final = refine_with_llm(draft, refiner_llm, goals=[...])
```

### Cost Optimization Strategy

**Hybrid Approach**:
1. Use **Gemini 2.0 Flash** for:
   - Multiple parallel data gathering tasks
   - Initial drafts/rough outputs
   - High-volume, simpler tasks

2. Use **GPT-4o** for:
   - Final refinement/polishing
   - Complex reasoning tasks
   - Quality-critical outputs

**Example**: Writing pipeline
```
[Gemini] Research (parallel) →
[Gemini] Initial draft →
[GPT-4o] Quality refinement →
[Gemini] Format/structure →
[GPT-4o] Final review
```

---

## Decision Framework

### Choose Google ADK When:

1. ✅ Multiple agents need to collaborate
2. ✅ Parallel execution is beneficial
3. ✅ Using Google Cloud services (Vertex AI, Google Search)
4. ✅ Need event streaming for real-time updates
5. ✅ Building complex, stateful workflows
6. ✅ Cost-sensitive (many API calls)

### Choose LangChain + OpenAI When:

1. ✅ Single task with quality criteria
2. ✅ Iterative refinement is primary pattern
3. ✅ Need GPT-4o's reasoning capability
4. ✅ Prompt engineering is critical
5. ✅ Simple, focused use case
6. ✅ Want procedural control flow

### Use Both When:

1. ✅ Complex system with both orchestration and refinement needs
2. ✅ Want cost optimization (Gemini for volume, GPT-4o for quality)
3. ✅ Different tasks have different requirements
4. ✅ Building production system with multiple stages

---

## Summary

| Aspect | Google ADK | LangChain + OpenAI | Hybrid |
|--------|------------|-------------------|--------|
| **Complexity** | High | Low | Medium |
| **Flexibility** | Agent composition | Prompt control | Both |
| **Cost** | Lower (Gemini) | Higher (GPT-4o) | Optimized |
| **Quality** | Good | Excellent | Best |
| **Use Case** | Multi-agent systems | Single-agent refinement | Complex systems |
| **Learning Curve** | Steeper | Gentler | Requires both |

**Recommendation**: Start with the framework that matches your primary use case, then add the other framework if needed for specific tasks.

---

**Next Steps**:
- Understand core concepts → [02_CORE_CONCEPTS.md](02_CORE_CONCEPTS.md)
- See specific patterns → [03-06] Pattern files
- Make architectural decisions → [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)
