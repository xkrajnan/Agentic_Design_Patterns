# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an educational repository containing Jupyter notebooks and code examples from "Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems" by Antonio Gulli. The repository demonstrates various agentic design patterns using Google's Agent Development Kit (ADK) and OpenAI's APIs.

## Structure

```
notebooks/          # Jupyter notebooks demonstrating agentic patterns
  Chapter 3_        # Parallelization examples
  Chapter 5_        # Tool Use patterns (Google Search, code execution, Vertex AI)
  Chapter 7_        # Multi-Agent Collaboration (coordinator, parallel, sequential, loops)
  Chapter 11_       # Goal Setting and Monitoring patterns
```

## Key Technologies

- **Google ADK (Agent Development Kit)**: Primary framework for building agents
  - `google.adk.agents`: Core agent classes (LlmAgent, BaseAgent, ParallelAgent, SequentialAgent)
  - `google.adk.tools`: Pre-built tools like `google_search`
  - `google.adk.runners`: Runner for executing agents
  - `google.adk.sessions`: Session management (InMemorySessionService)
  - Model: `gemini-2.0-flash-exp` (Gemini model)

- **LangChain + OpenAI**: Used in goal-setting patterns
  - `langchain_openai.ChatOpenAI`
  - Model: `gpt-4o`

## Common Agent Patterns

### 1. Parallel Execution (Chapter 3)
Multiple agents execute concurrently using `ParallelAgent`. Each sub-agent stores results in session state via `output_key`, then a merger agent synthesizes results using `SequentialAgent`.

### 2. Tool Use (Chapter 5)
Agents are configured with tools via the `tools` parameter. Pre-built tools include `google_search`. Custom tools can be created.

### 3. Multi-Agent Collaboration (Chapter 7)
- **Coordinator Pattern**: Parent agent with `sub_agents` delegates tasks
- **Sequential Pattern**: `SequentialAgent` runs agents in order
- **Parallel Pattern**: `ParallelAgent` runs agents concurrently
- **Custom Agents**: Extend `BaseAgent` and implement `_run_async_impl()`

### 4. Goal Setting and Monitoring (Chapter 11)
Iterative refinement loop: generate code → evaluate against goals → refine until goals are met or max iterations reached. Uses LLM as judge to determine goal completion.

## Agent Architecture

Key concepts when working with agents:
- Agents have hierarchical parent-child relationships via `sub_agents`
- Use `output_key` to store results in session state for later agents
- Custom agents must extend `BaseAgent` and implement `_run_async_impl` returning `AsyncGenerator[Event, None]`
- LlmAgent requires `name`, `model`, and typically `instruction` and `description`

## Environment Setup

Notebooks expect:
- `.env` file with API keys (OPENAI_API_KEY for Chapter 11)
- Google ADK setup following https://google.github.io/adk-docs/get-started/quickstart/
- Dependencies: `langchain_openai`, `openai`, `python-dotenv`, `nest_asyncio`

## Running Examples

All examples are in Jupyter notebooks. Execute cells sequentially. Some notebooks use `nest_asyncio.apply()` and `asyncio.run()` for async execution in notebook environments.

For Chapter 11 examples, generated code files are saved to the current working directory with pattern `{description}_{random_suffix}.py`.
