# Decision Frameworks: Pattern Selection and Architecture

## Overview

This document provides decision frameworks, performance matrices, and complementarity analyses to help select the right pattern for your use case.

---

## Pattern Selection Flowchart

```
START: What is your primary need?
│
├─ SINGLE TASK with quality criteria?
│  └─ Pattern: GOAL SETTING + LLM-AS-JUDGE
│     File: 06_GOAL_SETTING_ITERATION.md
│     Framework: LangChain + OpenAI
│
├─ MULTIPLE INDEPENDENT TASKS (same input)?
│  └─ Pattern: PARALLELAGENT
│     File: 03_PARALLELIZATION_PATTERNS.md
│     Framework: Google ADK
│
├─ MULTI-STEP PIPELINE (dependencies)?
│  └─ Pattern: SEQUENTIALAGENT
│     File: 03_PARALLELIZATION_PATTERNS.md
│     Framework: Google ADK
│
├─ ITERATIVE until condition met?
│  └─ Pattern: LOOPAGENT
│     File: 05_MULTIAGENT_COLLABORATION.md
│     Framework: Google ADK
│
├─ INTELLIGENT TASK ROUTING?
│  └─ Pattern: COORDINATOR
│     File: 05_MULTIAGENT_COLLABORATION.md
│     Framework: Google ADK
│
├─ MODULAR COMPOSITION?
│  └─ Pattern: AGENTTOOL
│     File: 05_MULTIAGENT_COLLABORATION.md
│     Framework: Google ADK
│
└─ NEED TOOLS (search, code execution, documents)?
   └─ Pattern: TOOL USE
      File: 04_TOOL_USE_PATTERNS.md
      Framework: Google ADK
```

---

## Performance Comparison Matrix

| Pattern | API Calls | Latency | Cost | Complexity | Best For |
|---------|-----------|---------|------|------------|----------|
| **Google Search Tool** | N (per search) | Medium | Low | Low | Web research |
| **Code Execution** | N + execution | Medium | Low | Low | Python code gen/exec |
| **Vertex AI Search** | N + search | Medium | Medium | Low | Document retrieval |
| **ParallelAgent (3 agents)** | 3N | min(t1,t2,t3) | 3× | Medium | Concurrent data gathering |
| **SequentialAgent (3 agents)** | 3N | t1+t2+t3 | 3× | Low | Data pipelines |
| **Coordinator** | Varies | Sequential | Varies | High | Task routing |
| **Loop (5 iter max)** | Up to 5N | Iterative | Up to 5× | Medium | Refinement loops |
| **AgentTool** | Depends | Depends | Depends | Low | Modular systems |
| **Goal Setting (5 iter)** | Up to 15 | Sequential | High | Medium | Quality-driven gen |

**Legend**:
- N = number of agents
- t1, t2, t3 = execution time per agent
- × = multiplier of single agent cost

---

## Complementarity Matrix

Patterns that combine synergistically:

| Pattern 1 | Pattern 2 | Synergy | Example Use Case |
|-----------|-----------|---------|------------------|
| **Parallel** | **Sequential** | ⭐⭐⭐ | Gather data in parallel, process sequentially |
| **Sequential** | **Goal Setting** | ⭐⭐ | Pipeline with quality refinement at end |
| **Coordinator** | **AgentTool** | ⭐⭐⭐ | Route to modular agent-tools |
| **Loop** | **Goal Setting** | ⭐⭐ | Iterative refinement with LLM-as-judge |
| **Parallel** | **Tool Use** | ⭐⭐⭐ | Multiple agents using tools concurrently |
| **Sequential** | **Loop** | ⭐⭐ | Pipeline with iterative step |
| **Any Pattern** | **Tool Use** | ⭐⭐⭐ | All agents can use tools |

⭐⭐⭐ = Highly complementary  
⭐⭐ = Complementary  
⭐ = Can combine but limited synergy

---

## Orthogonality Analysis

**Independent Dimensions** (can be mixed):

### Dimension 1: Execution Model
- **Parallel**: All agents run concurrently
- **Sequential**: Agents run in order
- **Iterative**: Agents loop until condition

*Choice depends on*: Task dependencies

### Dimension 2: Communication Model
- **State-based**: Via session.state and output_key
- **Event-based**: Via Event and AsyncGenerator
- **Tool-based**: Via tool calls and returns

*Choice depends on*: Data flow requirements

### Dimension 3: Decision Making
- **Rule-based**: Predefined logic (SequentialAgent)
- **LLM-based**: Intelligent routing (Coordinator)
- **Condition-based**: Check state (LoopAgent, Goal Setting)

*Choice depends on*: Complexity of routing logic

### Dimension 4: Tool Integration
- **Pre-built**: google_search, BuiltInCodeExecutor
- **Custom**: User-defined functions
- **Agent-as-tool**: AgentTool pattern

*Choice depends on*: Required capabilities

**Example Combination**:
- Sequential execution (Dim 1)
- State-based communication (Dim 2)
- LLM-based decisions (Dim 3)
- Custom tools (Dim 4)

→ Result: Sequential pipeline with intelligent agents using custom tools

---

## Architecture Trade-offs

### Google ADK vs LangChain/OpenAI

| Aspect | Google ADK | LangChain + OpenAI | Hybrid |
|--------|------------|-------------------|--------|
| **Multi-agent** | ✅ Native | ❌ Manual | ✅ Use ADK |
| **Concurrency** | ✅ Built-in | ❌ None | ✅ Use ADK |
| **Quality Refinement** | ⚠️ Via Loop | ✅ LLM-as-judge | ✅ Use both |
| **Prompt Control** | ⚠️ Less granular | ✅ Full control | ✅ Mix as needed |
| **Model Quality** | Good (Gemini) | Excellent (GPT-4o) | Best of both |
| **Cost** | Lower | Higher | Optimized |
| **Complexity** | Higher | Lower | Highest |

### When to Use Hybrid Approach

```python
# Use Gemini for data gathering (parallel, fast, cheap)
parallel_gatherer = ParallelAgent(sub_agents=[...])  # Google ADK

# Use GPT-4o for quality refinement (superior reasoning)
refiner = run_code_agent(use_case=..., llm=ChatOpenAI(model="gpt-4o"))  # LangChain
```

---

## Concurrency vs Sequential Guidance

### Choose Parallel When:
- ✅ Tasks are **independent**
- ✅ All need **same input**
- ✅ **Latency matters** (want speed)
- ✅ No inter-task dependencies

**Example**: Gathering weather + news + stocks simultaneously

### Choose Sequential When:
- ✅ Tasks have **dependencies**
- ✅ Output of task N → input of task N+1
- ✅ Building **data pipeline**
- ✅ Order matters

**Example**: Extract → Transform → Validate → Load

### Choose Hybrid When:
- ✅ Some tasks parallel, others sequential
- ✅ Want **optimized latency** for independent parts
- ✅ Need **dependencies** for later stages

**Example**: Parallel research → Sequential synthesis

---

## Tool vs Agent Delegation

| Need | Solution | Pattern |
|------|----------|---------|
| Call external API/function | **Tool** | Add to `tools=[]` |
| Generate & execute code | **Code Executor** | `code_executor=BuiltInCodeExecutor()` |
| Delegate to specialized agent | **Sub-agent** | Add to `sub_agents=[]` |
| Make agent reusable as tool | **AgentTool** | `agent_tool.AgentTool(agent)` |

---

## Iteration Count Optimization

### Goal Setting Pattern

| Scenario | Recommended max_iterations | Reason |
|----------|---------------------------|--------|
| Simple goals (1-2) | 2-3 | Quick convergence |
| Medium goals (3-4) | 3-5 (default) | Balanced quality/cost |
| Complex goals (5+) | 5-7 | More refinement needed |
| Strict criteria | 7-10 | May need many iterations |

**Cost Impact**:
- Each iteration = 3 API calls (generation + feedback + goals_met)
- 5 iterations = up to 15 calls
- Early termination reduces actual cost

### Loop Pattern

| Scenario | Recommended max_iterations | Reason |
|----------|---------------------------|--------|
| Polling/checking | 10-20 | May need many checks |
| Iterative refinement | 5-10 | Convergence expected |
| Convergence algorithm | 20-50 | Algorithm-dependent |

**Important**: Always set `max_iterations` to prevent infinite loops!

---

## Cost Optimization Strategies

### 1. Use Parallel for Independent Tasks
```python
# ❌ Sequential: 3× latency
for agent in agents:
    result = run(agent)

# ✅ Parallel: min latency
parallel = ParallelAgent(sub_agents=agents)
```

### 2. Hybrid Model Selection
```python
# Gemini for volume tasks (cheap)
gatherers = [LlmAgent(model="gemini-2.0-flash", ...) for _ in range(5)]

# GPT-4o for quality tasks (expensive but better)
refiner = ChatOpenAI(model="gpt-4o")
```

### 3. Early Termination
```python
# Goal Setting: terminates when goals met
for i in range(max_iterations):
    if goals_met(...):
        break  # Saves remaining iterations

# Loop: terminates on escalate
if condition_met:
    yield Event(actions=EventActions(escalate=True))
```

### 4. Reduce Iteration Count
```python
# Default: 5 iterations
run_code_agent(max_iterations=5)  # 15 API calls max

# Optimized: 3 iterations
run_code_agent(max_iterations=3)  # 9 API calls max
```

---

## Hybrid Pattern Combinations

### Example 1: Research & Quality Refinement

```
ParallelAgent (Gemini, data gathering)
    ↓
SequentialAgent (combine results)
    ↓
Goal Setting (GPT-4o, quality refinement)
```

### Example 2: Multi-Stage Pipeline

```
SequentialAgent [
    ParallelAgent (stage 1: gather data),
    LlmAgent (stage 2: process),
    LoopAgent (stage 3: iterative refinement)
]
```

### Example 3: Modular System

```
Coordinator
    ├─→ AgentTool(ResearchAgent) [uses google_search]
    ├─→ AgentTool(CodeAgent) [uses BuiltInCodeExecutor]
    └─→ AgentTool(AnalysisAgent) [runs Goal Setting loop]
```

---

## Summary Decision Table

| Your Need | Pattern | File | Complexity |
|-----------|---------|------|------------|
| Concurrent data gathering | ParallelAgent | 03 | Medium |
| Data transformation pipeline | SequentialAgent | 03 | Low |
| Both above combined | Hybrid | 03 | Medium |
| Web search | Google Search Tool | 04 | Low |
| Code generation + execution | Code Execution Tool | 04 | Low |
| Document search | Vertex AI Search | 04 | Low |
| Intelligent task routing | Coordinator | 05 | High |
| Iterative refinement | LoopAgent | 05 | Medium |
| Modular composition | AgentTool | 05 | Low |
| Quality-driven generation | Goal Setting | 06 | Medium |

**Next**: Troubleshoot issues → [08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)
