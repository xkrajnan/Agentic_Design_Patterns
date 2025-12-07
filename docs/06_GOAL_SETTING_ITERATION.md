# Goal Setting and Iterative Refinement

## Overview

Chapter 11 demonstrates the **Iterative Refinement Pattern** using LangChain + OpenAI:
- LLM-as-judge for quality evaluation
- Two-tier evaluation (detailed feedback + binary decision)
- Iterative code generation until goals met

**Source**: Chapter 11 - Goal Setting and Monitoring

---

## Complete Runnable Example

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import random
import re
from pathlib import Path

# Setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ Set OPENAI_API_KEY environment variable")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=OPENAI_API_KEY)

# === Utility Functions ===

def generate_prompt(use_case: str, goals: list[str], previous_code: str = "", feedback: str = "") -> str:
    """Construct LLM prompt for code generation."""
    prompt = f"""You are an AI coding agent. Write Python code for: {use_case}

Your goals are:
{chr(10).join(f"- {g}" for g in goals)}
"""
    if previous_code:
        prompt += f"\\n\\nPreviously generated code:\\n{previous_code}"
    if feedback:
        prompt += f"\\n\\nFeedback on previous version:\\n{feedback}"
    prompt += "\\n\\nReturn only the revised Python code. No explanations."
    return prompt

def get_code_feedback(code: str, goals: list[str]) -> str:
    """Request detailed code critique from LLM."""
    feedback_prompt = f"""Review this code against these goals:
{chr(10).join(f"- {g}" for g in goals)}

Code:
{code}

Critique: identify if goals are met and needed improvements."""
    return llm.invoke(feedback_prompt)

def goals_met(feedback_text: str, goals: list[str]) -> bool:
    """Binary judgment: did code meet goals?"""
    review_prompt = f"""Goals: {chr(10).join(f"- {g}" for g in goals)}

Feedback: {feedback_text}

Have the goals been met? Respond with only: True or False."""
    response = llm.invoke(review_prompt).content.strip().lower()
    return response == "true"

def clean_code_block(code: str) -> str:
    """Parse code from markdown-wrapped format."""
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\\n".join(lines).strip()

def add_comment_header(code: str, use_case: str) -> str:
    """Prepend use-case comment header."""
    return f"# This Python program implements: {use_case}\\n\\n{code}"

def save_code_to_file(code: str, use_case: str) -> str:
    """Save code with smart filename generation."""
    # Get LLM-generated summary for filename
    summary_prompt = f"Summarize to max 10 chars for filename: {use_case}"
    raw_summary = llm.invoke(summary_prompt).content.strip()
    
    # Sanitize
    short_name = re.sub(r"[^a-zA-Z0-9_]", "", raw_summary.replace(" ", "_").lower())[:10]
    random_suffix = str(random.randint(1000, 9999))
    filename = f"{short_name}_{random_suffix}.py"
    filepath = Path.cwd() / filename
    
    with open(filepath, "w") as f:
        f.write(code)
    
    print(f"✅ Code saved to: {filepath}")
    return str(filepath)

# === Main Agent Function ===

def run_code_agent(use_case: str, goals_input: str, max_iterations: int = 5) -> str:
    """Main iterative refinement loop."""
    goals = [g.strip() for g in goals_input.split(",")]
    
    print(f"\\n🎯 Use Case: {use_case}")
    print("🎯 Goals:", ", ".join(goals))
    
    previous_code = ""
    feedback = ""
    
    for i in range(max_iterations):
        print(f"\\n=== 🔁 Iteration {i + 1} of {max_iterations} ===")
        
        # Generate/refine code
        prompt = generate_prompt(use_case, goals, previous_code, 
                                feedback.content if feedback else "")
        code_response = llm.invoke(prompt)
        code = clean_code_block(code_response.content.strip())
        print(f"\\n🧾 Generated Code:\\n{'-'*50}\\n{code}\\n{'-'*50}")
        
        # Get feedback
        feedback = get_code_feedback(code, goals)
        feedback_text = feedback.content.strip()
        print(f"\\n📥 Feedback:\\n{'-'*50}\\n{feedback_text}\\n{'-'*50}")
        
        # Check if goals met
        if goals_met(feedback_text, goals):
            print("✅ LLM confirms goals are met. Stopping.")
            break
        
        print("🛠️ Goals not fully met. Continuing...")
        previous_code = code
    
    # Finalize
    final_code = add_comment_header(code, use_case)
    return save_code_to_file(final_code, use_case)

# === Usage Example ===

if __name__ == "__main__":
    filepath = run_code_agent(
        use_case="Write code to find BinaryGap of a given positive integer",
        goals_input="Simple to understand, Functionally correct, Handles edge cases, Prints examples"
    )
    print(f"\\n✨ Complete! File: {filepath}")
```

---

## Key Patterns

### 1. LLM-as-Judge (Two-Tier Evaluation)

**Tier 1 - Detailed Feedback**:
```python
feedback = get_code_feedback(code, goals)  # Detailed critique
```

**Tier 2 - Binary Decision**:
```python
is_done = goals_met(feedback_text, goals)  # True/False only
```

**Why Two Tiers?**
- Detailed feedback → actionable improvements
- Binary decision → reliable loop termination

### 2. Iterative Refinement Loop

```
for iteration in 1..max_iterations:
    1. Generate code (using previous + feedback)
    2. Get detailed feedback
    3. Check if goals met (binary)
    4. If met → break, else → continue
```

### 3. State Preservation

```python
previous_code = ""      # Starts empty
feedback = ""           # Starts empty

# Each iteration:
previous_code = code    # Update for next iteration
feedback = new_feedback # Update for next iteration
```

### 4. Prompt Engineering

**Conditional Context Injection**:
```python
if previous_code:
    prompt += f"\\nPreviously: {previous_code}"
if feedback:
    prompt += f"\\nFeedback: {feedback}"
```

---

## Performance Considerations

### Temperature Setting

```python
llm = ChatOpenAI(temperature=0.3)  # Low = deterministic
```

**Why Low Temperature?**
- Need consistent refinements across iterations
- Code generation should be technical, not creative
- Helps convergence to goal criteria

### Iteration Count

| Goals Complexity | Recommended max_iterations |
|-----------------|---------------------------|
| Simple (1-2 goals) | 2-3 iterations |
| Medium (3-4 goals) | 3-5 iterations (default) |
| Complex (5+ goals) | 5-7 iterations |

### API Call Cost

```python
# Per iteration:
# - 1 call for code generation
# - 1 call for feedback
# - 1 call for goals_met check
# Total: 3 calls per iteration

# With max_iterations=5:
# Maximum: 15 API calls
# Typical (early termination): 6-9 calls
```

---

## Dynamic Filename Generation

```python
# Step 1: LLM generates semantic name
summary = llm.invoke("Summarize for filename: {use_case}")

# Step 2: Sanitize
clean_name = re.sub(r"[^a-zA-Z0-9_]", "", summary)[:10]

# Step 3: Add random suffix
filename = f"{clean_name}_{random.randint(1000, 9999)}.py"

# Result: binarygap_7268.py (meaningful + unique)
```

---

## Decision Framework

**Use this pattern when**:
- ✅ Single task with quality criteria
- ✅ Clear evaluation goals
- ✅ Iterative improvement beneficial
- ✅ Code/content generation task

**Don't use when**:
- ❌ Need multi-agent coordination (use Google ADK)
- ❌ No clear goals to evaluate against
- ❌ Single-shot generation sufficient

---

## Comparison with Google ADK Loop

| Aspect | LangChain Goal Setting | Google ADK LoopAgent |
|--------|----------------------|---------------------|
| **Framework** | LangChain + OpenAI | Google ADK |
| **Termination** | LLM judges goals met | EventActions(escalate=True) |
| **State** | Python variables | Session state |
| **Complexity** | Simple (functions) | Higher (agents) |
| **Flexibility** | High (procedural) | Medium (agent-based) |

**Next**: See decision frameworks → [07_DECISION_FRAMEWORKS.md](07_DECISION_FRAMEWORKS.md)
