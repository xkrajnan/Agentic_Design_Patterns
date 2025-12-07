# Parallelization Patterns: ParallelAgent & SequentialAgent

## Overview

This document covers the two core patterns for orchestrating multiple agents in Google ADK:

1. **ParallelAgent**: Execute multiple agents concurrently on the same input
2. **SequentialAgent**: Execute agents in order, passing results through a pipeline
3. **Hybrid Pattern**: Combining parallel and sequential execution

**Source**: Chapter 3 - Parallelization (Google ADK Code Example)

---

## 1. ParallelAgent Pattern

### What Is ParallelAgent?

**ParallelAgent** orchestrates the **concurrent execution** of multiple sub-agents. All sub-agents:
- Receive the **same input** message
- Execute **simultaneously** (true parallelism)
- Store results **independently** in session state via `output_key`
- Complete before the ParallelAgent finishes

### When to Use

✅ **Use ParallelAgent when**:
- Gathering data from multiple independent sources
- Multiple tasks can run concurrently without dependencies
- All tasks need the same input
- Want to minimize total execution time (latency)

❌ **Don't use when**:
- Tasks have dependencies (use SequentialAgent)
- Results from one task needed by another before completion
- Single task is sufficient

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent, ParallelAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import asyncio
import nest_asyncio

# Enable async in Jupyter
nest_asyncio.apply()

# Configuration
APP_NAME = \"parallel_research\"
USER_ID = \"user123\"
SESSION_ID = \"session456\"
MODEL = \"gemini-2.0-flash-exp\"

# Define 3 parallel research agents
researcher_1 = LlmAgent(
    name=\"RenewableEnergyResearcher\",
    model=MODEL,
    instruction=\"\"\"You are an AI Research Assistant specializing in energy.
Research the latest advancements in 'renewable energy sources'.
Use the Google Search tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.\"\"\",
    description=\"Researches renewable energy sources.\",
    tools=[google_search],
    output_key=\"renewable_energy_result\"  # Results stored here
)

researcher_2 = LlmAgent(
    name=\"EVResearcher\",
    model=MODEL,
    instruction=\"\"\"You are an AI Research Assistant specializing in transportation.
Research the latest developments in 'electric vehicle technology'.
Use the Google Search tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.\"\"\",
    description=\"Researches electric vehicle technology.\",
    tools=[google_search],
    output_key=\"ev_technology_result\"
)

researcher_3 = LlmAgent(
    name=\"CarbonCaptureResearcher\",
    model=MODEL,
    instruction=\"\"\"You are an AI Research Assistant specializing in climate solutions.
Research the current state of 'carbon capture methods'.
Use the Google Search tool provided.
Summarize your key findings concisely (1-2 sentences).
Output *only* the summary.\"\"\",
    description=\"Researches carbon capture methods.\",
    tools=[google_search],
    output_key=\"carbon_capture_result\"
)

# Create ParallelAgent
parallel_research_agent = ParallelAgent(
    name=\"ParallelWebResearchAgent\",
    sub_agents=[researcher_1, researcher_2, researcher_3],
    description=\"Runs multiple research agents in parallel to gather information.\"
)

# Execute
async def run_parallel_demo():
    # Create session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Create runner
    runner = Runner(
        agent=parallel_research_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Prepare message
    content = types.Content(
        role='user',
        parts=[types.Part(text=\"Research sustainable technology trends\")]
    )

    # Execute and collect results
    print(\"Starting parallel research...\\n\")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.author:
            print(f\"[{event.author}] Working...\")

    # Access results from session state
    final_session = runner.get_session()
    print(\"\\n=== Results ===\")
    print(f\"Renewable Energy: {final_session.state.get('renewable_energy_result')}\")
    print(f\"Electric Vehicles: {final_session.state.get('ev_technology_result')}\")
    print(f\"Carbon Capture: {final_session.state.get('carbon_capture_result')}\")

# Run
asyncio.run(run_parallel_demo())
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Execution** | All sub-agents run simultaneously |
| **Input** | Same message sent to all sub-agents |
| **Output** | Each agent stores result via `output_key` |
| **Completion** | Waits for ALL agents to finish |
| **State** | Results isolated by unique keys |
| **Dependencies** | No inter-agent dependencies |

### State Management in ParallelAgent

```
Input Message: \"Research topic X\"
    ↓
ParallelAgent distributes to all sub-agents
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Agent 1        │  Agent 2        │  Agent 3        │
│  (concurrent)   │  (concurrent)   │  (concurrent)   │
└─────────────────┴─────────────────┴─────────────────┘
    ↓                   ↓                   ↓
session.state[\"key1\"]  session.state[\"key2\"]  session.state[\"key3\"]
    ↓
ParallelAgent completes when ALL agents finish
```

### Performance Considerations

```python
# Sequential execution time (without ParallelAgent)
# Time = Agent1_time + Agent2_time + Agent3_time
# Example: 3 seconds + 3 seconds + 3 seconds = 9 seconds total

# Parallel execution time (with ParallelAgent)
# Time = max(Agent1_time, Agent2_time, Agent3_time)
# Example: max(3, 3, 3) = 3 seconds total

# Speedup = 3x in this case
```

**API Call Costs**:
- 3 agents in parallel = 3 simultaneous API calls
- Cost = 3x single agent cost
- Latency = 1x (not 3x)

---

## 2. SequentialAgent Pattern

### What Is SequentialAgent?

**SequentialAgent** orchestrates the **sequential execution** of multiple sub-agents. Agents:
- Execute in **strict order** (agent N+1 waits for agent N)
- Can **pass data** via session state
- Build upon **previous results**
- Create data transformation **pipelines**

### When to Use

✅ **Use SequentialAgent when**:
- Tasks have dependencies (output of A → input of B)
- Building data transformation pipelines (ETL)
- Need to accumulate context across steps
- Order of execution matters

❌ **Don't use when**:
- Tasks are independent (use ParallelAgent)
- Need parallel execution for speed
- Single agent sufficient

### Complete Runnable Example

```python
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Configuration
APP_NAME = \"sequential_pipeline\"
USER_ID = \"user123\"
SESSION_ID = \"session456\"
MODEL = \"gemini-2.0-flash-exp\"

# Step 1: Extract data
extractor = LlmAgent(
    name=\"DataExtractor\",
    model=MODEL,
    instruction=\"\"\"Extract the key entities and facts from the user's message.
Format as a bullet list.
Output only the extracted data.\"\"\",
    description=\"Extracts structured data from text\",
    output_key=\"extracted_data\"  # Stores result for next agent
)

# Step 2: Transform data
transformer = LlmAgent(
    name=\"DataTransformer\",
    model=MODEL,
    instruction=\"\"\"Transform the extracted data into a structured format.

Extracted Data:
{extracted_data}

Convert to JSON format with keys: entities, facts, topics.
Output only the JSON.\"\"\",
    description=\"Transforms data to JSON\",
    output_key=\"transformed_data\"
)

# Step 3: Validate data
validator = LlmAgent(
    name=\"DataValidator\",
    model=MODEL,
    instruction=\"\"\"Validate the transformed data for completeness and correctness.

Transformed Data:
{transformed_data}

Check:
1. JSON is well-formed
2. All required keys present
3. Data is logically consistent

Output: \"VALID\" or \"INVALID: <reason>\"\"\"\",
    description=\"Validates transformed data\",
    output_key=\"validation_result\"
)

# Step 4: Summarize process
summarizer = LlmAgent(
    name=\"ProcessSummarizer\",
    model=MODEL,
    instruction=\"\"\"Summarize the entire ETL process results:

Extracted: {extracted_data}
Transformed: {transformed_data}
Validation: {validation_result}

Provide a 2-sentence summary of the process outcome.\"\"\",
    description=\"Summarizes the pipeline results\"
)

# Create SequentialAgent pipeline
etl_pipeline = SequentialAgent(
    name=\"ETL_Pipeline\",
    sub_agents=[
        extractor,      # Step 1
        transformer,    # Step 2 (uses Step 1 output)
        validator,      # Step 3 (uses Step 2 output)
        summarizer      # Step 4 (uses all previous outputs)
    ],
    description=\"Extract-Transform-Load pipeline\"
)

# Execute
async def run_sequential_demo():
    # Create session
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # Create runner
    runner = Runner(
        agent=etl_pipeline,
        app_name=APP_NAME,
        session_service=session_service
    )

    # Prepare message
    content = types.Content(
        role='user',
        parts=[types.Part(text=\"\"\"The Eiffel Tower in Paris, France was built in 1889.
It stands 330 meters tall and was designed by engineer Gustave Eiffel.
The tower receives about 7 million visitors annually.\"\"\")]
    )

    # Execute pipeline
    print(\"Starting ETL pipeline...\\n\")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.author:
            print(f\"[{event.author}] Processing...\")
        if event.is_final_response():
            print(\"\\n=== Final Summary ===\")
            print(event.content.parts[0].text)

    # Show intermediate results
    final_session = runner.get_session()
    print(\"\\n=== Pipeline State ===\")
    print(f\"Extracted: {final_session.state.get('extracted_data')}\")
    print(f\"Transformed: {final_session.state.get('transformed_data')}\")
    print(f\"Validation: {final_session.state.get('validation_result')}\")

# Run
asyncio.run(run_sequential_demo())
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Execution** | One agent at a time, in order |
| **Input** | First agent gets user message, others use state |
| **Output** | Each agent can store via `output_key` |
| **Completion** | Completes when last agent finishes |
| **State** | Accumulates across pipeline |
| **Dependencies** | Strong: each step may depend on previous |

### Data Flow in SequentialAgent

```
User Input
    ↓
Agent 1 (Extractor)
    ↓ (stores in session.state[\"extracted_data\"])
Agent 2 (Transformer) ← reads {extracted_data}
    ↓ (stores in session.state[\"transformed_data\"])
Agent 3 (Validator) ← reads {transformed_data}
    ↓ (stores in session.state[\"validation_result\"])
Agent 4 (Summarizer) ← reads {extracted_data}, {transformed_data}, {validation_result}
    ↓
Final Output
```

---

## 3. Hybrid Pattern: Parallel + Sequential

### What Is the Hybrid Pattern?

Combine **ParallelAgent** and **SequentialAgent** to:
1. **Gather data in parallel** (fast)
2. **Process sequentially** (dependencies)

This is the pattern demonstrated in Chapter 3.

### When to Use

✅ **Use Hybrid Pattern when**:
- Initial data gathering is independent (parallel)
- Subsequent processing requires combined results (sequential)
- Want to optimize both latency AND dependencies

### Complete Runnable Example (from Chapter 3)

```python
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import asyncio
import nest_asyncio

nest_asyncio.apply()

# Configuration
APP_NAME = \"hybrid_pipeline\"
USER_ID = \"user123\"
SESSION_ID = \"session456\"
MODEL = \"gemini-2.0-flash-exp\"

# === PHASE 1: PARALLEL DATA GATHERING ===

researcher_1 = LlmAgent(
    name=\"RenewableEnergyResearcher\",
    model=MODEL,
    instruction=\"\"\"Research renewable energy sources using google_search.
Summarize in 1-2 sentences. Output only the summary.\"\"\",
    tools=[google_search],
    output_key=\"renewable_energy_result\"
)

researcher_2 = LlmAgent(
    name=\"EVResearcher\",
    model=MODEL,
    instruction=\"\"\"Research electric vehicle technology using google_search.
Summarize in 1-2 sentences. Output only the summary.\"\"\",
    tools=[google_search],
    output_key=\"ev_technology_result\"
)

researcher_3 = LlmAgent(
    name=\"CarbonCaptureResearcher\",
    model=MODEL,
    instruction=\"\"\"Research carbon capture methods using google_search.
Summarize in 1-2 sentences. Output only the summary.\"\"\",
    tools=[google_search],
    output_key=\"carbon_capture_result\"
)

# Parallel orchestrator
parallel_research = ParallelAgent(
    name=\"ParallelWebResearchAgent\",
    sub_agents=[researcher_1, researcher_2, researcher_3]
)

# === PHASE 2: SEQUENTIAL SYNTHESIS ===

merger = LlmAgent(
    name=\"SynthesisAgent\",
    model=MODEL,
    instruction=\"\"\"Synthesize research findings into a structured report.

**Input Summaries:**

*   **Renewable Energy:**
    {renewable_energy_result}

*   **Electric Vehicles:**
    {ev_technology_result}

*   **Carbon Capture:**
    {carbon_capture_result}

**Output Format:**

## Summary of Recent Sustainable Technology Advancements

### Renewable Energy Findings
[Elaborate on renewable energy summary]

### Electric Vehicle Findings
[Elaborate on EV summary]

### Carbon Capture Findings
[Elaborate on carbon capture summary]

### Overall Conclusion
[1-2 sentence conclusion connecting the findings]

Output only the structured report.\"\"\",
    description=\"Combines findings into structured report\"
)

# === HYBRID PIPELINE: Parallel → Sequential ===

hybrid_pipeline = SequentialAgent(
    name=\"ResearchAndSynthesisPipeline\",
    sub_agents=[
        parallel_research,  # Phase 1: Gather in parallel
        merger               # Phase 2: Synthesize sequentially
    ],
    description=\"Coordinates parallel research and synthesizes results\"
)

# Execute
async def run_hybrid_demo():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    runner = Runner(
        agent=hybrid_pipeline,
        app_name=APP_NAME,
        session_service=session_service
    )

    content = types.Content(
        role='user',
        parts=[types.Part(text=\"Research sustainable technology trends and create a report\")]
    )

    print(\"Phase 1: Parallel Research (concurrent)...\\n\")
    print(\"Phase 2: Sequential Synthesis (after parallel completes)...\\n\")

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content
    ):
        if event.is_final_response():
            print(\"=== Final Report ===\")
            print(event.content.parts[0].text)

# Run
asyncio.run(run_hybrid_demo())
```

### Hybrid Pattern Execution Flow

```
Input: \"Research sustainable tech and create report\"
    ↓
SequentialAgent starts
    ↓
[PHASE 1] ParallelAgent executes
    ↓
    ├─→ Researcher 1 (concurrent) → output_key=\"renewable_energy_result\"
    ├─→ Researcher 2 (concurrent) → output_key=\"ev_technology_result\"
    └─→ Researcher 3 (concurrent) → output_key=\"carbon_capture_result\"
    ↓
    Wait for ALL parallel agents to complete
    ↓
[PHASE 2] Merger Agent executes (sequential)
    ↓
    Reads: {renewable_energy_result}, {ev_technology_result}, {carbon_capture_result}
    ↓
    Synthesizes into final report
    ↓
Output: Structured report
```

### Performance Analysis

```python
# Without parallelization (all sequential)
total_time = researcher1_time + researcher2_time + researcher3_time + merger_time
# Example: 3s + 3s + 3s + 2s = 11 seconds

# With parallelization (hybrid)
total_time = max(researcher1_time, researcher2_time, researcher3_time) + merger_time
# Example: max(3s, 3s, 3s) + 2s = 5 seconds

# Speedup: 11s → 5s (2.2x faster)
```

---

## Common Pitfalls

### 1. Missing output_key in ParallelAgent

```python
# ❌ WRONG - No output_key
ParallelAgent(sub_agents=[
    LlmAgent(name=\"A\", ...),  # Missing output_key!
    LlmAgent(name=\"B\", ...)
])

# ✅ CORRECT - All parallel agents have output_keys
ParallelAgent(sub_agents=[
    LlmAgent(name=\"A\", ..., output_key=\"result_a\"),
    LlmAgent(name=\"B\", ..., output_key=\"result_b\")
])
```

### 2. Template Variable Mismatch

```python
# ❌ WRONG - Mismatch between output_key and template variable
LlmAgent(..., output_key=\"my_data\")
LlmAgent(instruction=\"Use {mydata}\")  # Wrong name!

# ✅ CORRECT - Exact match
LlmAgent(..., output_key=\"my_data\")
LlmAgent(instruction=\"Use {my_data}\")  # Correct!
```

### 3. Using ParallelAgent When Sequential Needed

```python
# ❌ WRONG - These tasks have dependencies!
ParallelAgent(sub_agents=[
    extract_agent,    # Extracts data
    transform_agent   # Needs extracted data - can't run in parallel!
])

# ✅ CORRECT - Use Sequential for dependencies
SequentialAgent(sub_agents=[
    extract_agent,
    transform_agent
])
```

---

## Decision Matrix

| Scenario | Pattern | Reason |
|----------|---------|--------|
| Gather data from 3 APIs | **Parallel** | Independent, concurrent |
| Extract → Transform → Load | **Sequential** | Dependencies between steps |
| Research + Synthesize | **Hybrid** | Research parallel, synthesis sequential |
| Multiple independent calculations | **Parallel** | No dependencies, maximize speed |
| Multi-step validation | **Sequential** | Each step validates previous |
| Data gathering + Analysis | **Hybrid** | Gather parallel, analyze with all data |

---

## Summary

| Pattern | Execution | Use Case | Complexity | API Costs |
|---------|-----------|----------|------------|-----------|
| **ParallelAgent** | Concurrent | Independent data gathering | Medium | N × single cost |
| **SequentialAgent** | Sequential | Data pipelines with dependencies | Low | N × single cost |
| **Hybrid** | Mixed | Complex workflows | Medium-High | Optimized |

**Next Steps**:
- See tool integration → [04_TOOL_USE_PATTERNS.md](04_TOOL_USE_PATTERNS.md)
- See multi-agent patterns → [05_MULTIAGENT_COLLABORATION.md](05_MULTIAGENT_COLLABORATION.md)
- Make decisions → [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)
