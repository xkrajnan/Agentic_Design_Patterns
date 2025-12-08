# Agentic Design Patterns: LLM Agent Roadmap

## Overview

This comprehensive roadmap provides deep implementation guidance for LLM coding agents working with agentic design patterns. The repository demonstrates **8 distinct patterns** across **two frameworks** (Google ADK and LangChain/OpenAI) using real-world examples from Antonio Gulli's book.

## Documentation Structure

| File | Purpose | Key Content |
|------|---------|-------------|
| **[01_FRAMEWORKS_COMPARISON.md](01_FRAMEWORKS_COMPARISON.md)** | Framework deep dive | Google ADK vs LangChain/OpenAI, model differences, when to use each |
| **[02_CORE_CONCEPTS.md](02_CORE_CONCEPTS.md)** | Foundation concepts | Session state, events, parent-child relationships, tools |
| **[03_PARALLELIZATION_PATTERNS.md](03_PARALLELIZATION_PATTERNS.md)** | Parallel execution | ParallelAgent, SequentialAgent, output_key state management |
| **[04_TOOL_USE_PATTERNS.md](04_TOOL_USE_PATTERNS.md)** | Tool integration | Google Search, Code Execution, Vertex AI Search patterns |
| **[05_MULTIAGENT_COLLABORATION.md](05_MULTIAGENT_COLLABORATION.md)** | Multi-agent patterns | Coordinator, Parallel, Sequential, Loop, AgentTool (5 patterns) |
| **[06_GOAL_SETTING_ITERATION.md](06_GOAL_SETTING_ITERATION.md)** | Iterative refinement | LLM-as-judge, two-tier evaluation, prompt engineering |
| **[07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)** | Selection guidance | Flowcharts, performance matrices, complementarity analysis |
| **[08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)** | Problem solving | Common errors, solutions, debugging strategies |
| **[09_QUICK_REFERENCE.md](09_QUICK_REFERENCE.md)** | Cheat sheets | Imports, signatures, templates, quick lookup |
| **[10_UNIFIED_THEORY.md](10_UNIFIED_THEORY.md)** | Formal framework | 4D design space, composition algebra, theorems |
| **[11_AEROSPACE_RELIABILITY_PATTERNS.md](11_AEROSPACE_RELIABILITY_PATTERNS.md)** | Fault tolerance | FDIR, ECSS E1-E4, retry, TMR, circuit breaker |

## Quick Pattern Selector

Use this decision tree to quickly identify the right pattern for your task:

```
START: What do you need to accomplish?
│
├─ QUESTION 1: Do you need multiple agents?
│  │
│  ├─ NO → Use single LlmAgent with tools
│  │         See: 04_TOOL_USE_PATTERNS.md
│  │
│  └─ YES → Continue to QUESTION 2
│
├─ QUESTION 2: How should agents execute?
│  │
│  ├─ CONCURRENTLY (same input to all)
│  │  └─ Pattern: PARALLEL
│  │     File: 03_PARALLELIZATION_PATTERNS.md
│  │     Use: Data gathering from multiple sources simultaneously
│  │
│  ├─ SEQUENTIALLY (output → input chain)
│  │  └─ Pattern: SEQUENTIAL
│  │     File: 03_PARALLELIZATION_PATTERNS.md
│  │     Use: Data transformation pipelines (ETL)
│  │
│  ├─ ITERATIVELY (until condition met)
│  │  └─ Pattern: LOOP
│  │     File: 05_MULTIAGENT_COLLABORATION.md
│  │     Use: Refinement loops, polling, convergence tasks
│  │
│  └─ INTELLIGENTLY (parent decides which child)
│     └─ Pattern: COORDINATOR
│        File: 05_MULTIAGENT_COLLABORATION.md
│        Use: Task routing, specialized agent delegation
│
├─ QUESTION 3: Do you need modular composition?
│  │
│  └─ YES → Pattern: AGENTTOOL
│     File: 05_MULTIAGENT_COLLABORATION.md
│     Use: Agent-as-tool, reusable components, hierarchies
│
├─ QUESTION 4: Do you need iterative quality improvement?
│  │
│  └─ YES → Pattern: GOAL SETTING + LLM-AS-JUDGE
│     File: 06_GOAL_SETTING_ITERATION.md
│     Use: Code generation, content creation with criteria
│
├─ QUESTION 5: Do you need to search, execute code, or access data?
   │
   ├─ SEARCH WEB → Pattern: GOOGLE SEARCH TOOL
   │  File: 04_TOOL_USE_PATTERNS.md
   │
   ├─ EXECUTE CODE → Pattern: CODE EXECUTION
   │  File: 04_TOOL_USE_PATTERNS.md
   │
   └─ SEARCH DOCUMENTS → Pattern: VERTEX AI SEARCH
      File: 04_TOOL_USE_PATTERNS.md
│
└─ QUESTION 6: Do you need fault tolerance?
   │
   ├─ TRANSIENT FAILURES (API timeouts, rate limits)
   │  └─ Pattern: RETRY + CIRCUIT BREAKER
   │     File: 11_AEROSPACE_RELIABILITY_PATTERNS.md
   │
   ├─ CRITICAL DECISIONS (need consensus)
   │  └─ Pattern: TMR (Triple Modular Redundancy)
   │     File: 11_AEROSPACE_RELIABILITY_PATTERNS.md
   │
   ├─ SERVICE OUTAGES (need fallbacks)
   │  └─ Pattern: FALLBACK CHAIN
   │     File: 11_AEROSPACE_RELIABILITY_PATTERNS.md
   │
   └─ MISSION-CRITICAL (full protection)
      └─ Pattern: FDIR (Full Fault Detection, Isolation, Recovery)
         File: 11_AEROSPACE_RELIABILITY_PATTERNS.md
```

## Pattern Summary Table

| Pattern Name | Execution Model | Concurrency | Use Case | Complexity | File |
|--------------|----------------|-------------|----------|------------|------|
| **Google Search Tool** | Single agent + tool | No | Web information retrieval | Low | 04 |
| **Code Execution** | Single agent + executor | No | Python code generation & execution | Low | 04 |
| **Vertex AI Search** | Single specialized agent | No | Document/datastore queries | Low | 04 |
| **ParallelAgent** | Multiple agents | Yes | Concurrent data gathering | Medium | 03 |
| **SequentialAgent** | Multiple agents | No | Linear data pipelines | Low | 03 |
| **Coordinator** | Hierarchical agents | No | Intelligent task routing | High | 05 |
| **Loop** | Iterative agents | No | Refinement until condition | Medium | 05 |
| **AgentTool** | Agent-as-tool | Depends | Modular composition | Low | 05 |
| **Goal Setting** | Iterative single agent | No | Quality-driven refinement | Medium | 06 |
| **RetryAgent** | Wrapper | No | Transient failure handling | Low | 11 |
| **CircuitBreaker** | Wrapper | No | Cascade failure prevention | Low | 11 |
| **TMR (Triple Modular Redundancy)** | Redundant parallel | Yes | Critical decision consensus | Medium | 11 |
| **FallbackChain** | Fallback sequential | No | Graceful degradation | Medium | 11 |
| **FDIRAgent** | Full pipeline | Varies | Mission-critical protection | High | 11 |

## Getting Started

### For First-Time Users

1. **Start with Core Concepts**: Read [02_CORE_CONCEPTS.md](02_CORE_CONCEPTS.md) to understand:
   - Session state management
   - Event handling patterns
   - Parent-child agent relationships
   - Tool integration basics

2. **Choose Your Framework**: Read [01_FRAMEWORKS_COMPARISON.md](01_FRAMEWORKS_COMPARISON.md) to decide:
   - Google ADK (Gemini models, agent orchestration)
   - LangChain/OpenAI (GPT-4o, iterative refinement)

3. **Select a Pattern**: Use the Quick Pattern Selector above or browse [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)

4. **Copy & Adapt Code**: Each pattern file includes complete, runnable examples with all imports

5. **Troubleshoot**: If you encounter issues, see [08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)

### For Experienced Users

- **Quick Lookup**: Use [09_QUICK_REFERENCE.md](09_QUICK_REFERENCE.md) for imports, signatures, and templates
- **Decision Support**: Use [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md) for performance comparisons and pattern combinations
- **Advanced Patterns**: See complementarity matrices in [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md) for combining patterns

## Framework Quick Reference

### Google ADK (Chapters 3, 5, 7)

**Model**: `gemini-2.0-flash-exp` or `gemini-2.0-flash`

**Key Classes**:
- `LlmAgent` - Basic LLM-powered agent
- `ParallelAgent` - Concurrent execution orchestrator
- `SequentialAgent` - Sequential pipeline coordinator
- `LoopAgent` - Iterative execution controller
- `BaseAgent` - Abstract base for custom agents

**Execution**:
```python
from google.adk.runners import Runner
runner = Runner(agent=my_agent, session_service=session_service)
async for event in runner.run_async(user_id=..., session_id=..., new_message=...):
    # Process events
```

### LangChain + OpenAI (Chapter 11)

**Model**: `gpt-4o` (or `gpt-4-turbo`)

**Key Classes**:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
response = llm.invoke(prompt)
```

**Pattern**: Iterative refinement with LLM-as-judge

## Repository Structure

```
Agentic_Design_Patterns/
├── notebooks/
│   ├── Chapter 3_ Parallelization (Google ADK Code Example).ipynb
│   ├── Chapter 5_ Tool Use (using Google Search).ipynb
│   ├── Chapter 5_ Tool Use (Executing Code).ipynb
│   ├── Chapter 5_ Tool Use (Vertex AI Search).ipynb
│   ├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Coordinator).ipynb
│   ├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Parallel).ipynb
│   ├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Sequential).ipynb
│   ├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini Loop).ipynb
│   ├── Chapter 7_ Multi-Agent Collaboration - Code Example (ADK + Gemini AgentTooll).ipynb
│   └── Chapter 11_ Goal Setting and Monitoring (Goal_Setting_Iteration).ipynb
├── docs/          # ← You are here
│   ├── 00_ROADMAP_INDEX.md
│   ├── 01_FRAMEWORKS_COMPARISON.md
│   ├── 02_CORE_CONCEPTS.md
│   ├── 03_PARALLELIZATION_PATTERNS.md
│   ├── 04_TOOL_USE_PATTERNS.md
│   ├── 05_MULTIAGENT_COLLABORATION.md
│   ├── 06_GOAL_SETTING_ITERATION.md
│   ├── 07_DECISION_FRAMEWORKS.md
│   ├── 08_TROUBLESHOOTING.md
│   ├── 09_QUICK_REFERENCE.md
│   ├── 10_UNIFIED_THEORY.md
│   └── 11_AEROSPACE_RELIABILITY_PATTERNS.md
├── CLAUDE.md      # High-level overview
└── README.md      # Repository introduction
```

## Common Use Cases → Pattern Mapping

| Use Case | Recommended Pattern(s) | File |
|----------|----------------------|------|
| Gather data from multiple APIs concurrently | ParallelAgent | 03 |
| Transform data through multiple steps | SequentialAgent | 03 |
| Search web and summarize results | Google Search Tool + LlmAgent | 04 |
| Generate and execute Python code | Code Execution Tool | 04 |
| Search internal documents | Vertex AI Search | 04 |
| Route tasks to specialized agents | Coordinator | 05 |
| Refine output until quality criteria met | Loop or Goal Setting | 05, 06 |
| Build modular agent systems | AgentTool | 05 |
| Generate code meeting specific goals | Goal Setting + LLM-as-judge | 06 |
| Combine multiple patterns | See Complementarity Matrix | 07 |
| Handle transient API failures | RetryAgent + CircuitBreaker | 11 |
| Ensure decision consensus | TMR (Triple Modular Redundancy) | 11 |
| Graceful degradation on outages | FallbackChain | 11 |
| Mission-critical agent systems | FDIRAgent (full pipeline) | 11 |

## Pattern Relationships

### Complementary Patterns (Work Well Together)

- **ParallelAgent + SequentialAgent**: Gather data in parallel, then process sequentially
- **Coordinator + AgentTool**: Modular agents composed via intelligent coordinator
- **Loop + LLM-as-judge**: Iterative refinement with quality checking
- **Any pattern + Tools**: All agents can use tools for enhanced capabilities
- **Any pattern + RetryAgent**: Add fault tolerance to any agent
- **ParallelAgent + TMR**: Redundant execution with consensus voting
- **Goal Setting + FDIR**: Mission-critical code generation with full protection

### Orthogonal Dimensions (Independent Choices)

1. **D₁ Execution Model**: Parallel vs Sequential vs Iterative
2. **D₂ Communication**: State-based vs Event-based vs Tool-based
3. **D₃ Decision Making**: Rule-based vs LLM-based vs Condition-based
4. **D₄ Tool Integration**: Pre-built vs Custom vs Agent-as-tool
5. **D₅ Fault Tolerance**: None vs Retry vs Redundant vs FDIR

See [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md) for detailed analysis.
See [10_UNIFIED_THEORY.md](10_UNIFIED_THEORY.md) for formal 4D framework.
See [11_AEROSPACE_RELIABILITY_PATTERNS.md](11_AEROSPACE_RELIABILITY_PATTERNS.md) for 5D extension with fault tolerance.

## Next Steps

1. **Understand frameworks**: Start with [01_FRAMEWORKS_COMPARISON.md](01_FRAMEWORKS_COMPARISON.md)
2. **Learn core concepts**: Read [02_CORE_CONCEPTS.md](02_CORE_CONCEPTS.md)
3. **Pick a pattern**: Use the Quick Pattern Selector above
4. **Copy example code**: Each pattern file has complete runnable examples
5. **Troubleshoot if needed**: See [08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)

## Quick Links

- **Need to choose between ADK and LangChain?** → [01_FRAMEWORKS_COMPARISON.md](01_FRAMEWORKS_COMPARISON.md)
- **New to these concepts?** → [02_CORE_CONCEPTS.md](02_CORE_CONCEPTS.md)
- **Want performance comparisons?** → [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)
- **Having errors?** → [08_TROUBLESHOOTING.md](08_TROUBLESHOOTING.md)
- **Need quick code snippet?** → [09_QUICK_REFERENCE.md](09_QUICK_REFERENCE.md)
- **Want formal mathematical framework?** → [10_UNIFIED_THEORY.md](10_UNIFIED_THEORY.md)
- **Need fault tolerance / reliability?** → [11_AEROSPACE_RELIABILITY_PATTERNS.md](11_AEROSPACE_RELIABILITY_PATTERNS.md)

---

**Book Reference**: Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems by Antonio Gulli

**Repository**: https://github.com/yourusername/Agentic_Design_Patterns (update with actual URL)
