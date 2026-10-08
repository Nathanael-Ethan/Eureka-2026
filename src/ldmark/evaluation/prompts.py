"""
LDMARK Evaluation - Prompt Suite

Deterministic evaluation prompts covering various task categories.
All prompts are documented and free of politically sensitive content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class PromptCategory(Enum):
    """Categories of evaluation prompts."""
    FACTUAL_RECALL = "factual_recall"
    SIMPLE_REASONING = "simple_reasoning"
    ARITHMETIC = "arithmetic"
    CODE_COMPLETION = "code_completion"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class EvaluationPrompt:
    """A single evaluation prompt with metadata."""
    prompt: str
    category: PromptCategory
    expected_tokens: Optional[List[str]] = None  # For exact matching
    expected_answer: Optional[str] = None  # For classification/reasoning
    description: str = ""
    max_tokens: int = 50
    temperature: float = 0.0


# ============================================================================
# FACTUAL RECALL
# ============================================================================

FACTUAL_RECALL_PROMPTS = [
    EvaluationPrompt(
        prompt="The capital of France is",
        category=PromptCategory.FACTUAL_RECALL,
        expected_tokens=["Paris"],
        description="Basic geographic fact recall",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="Water boils at",
        category=PromptCategory.FACTUAL_RECALL,
        expected_tokens=["100", "degrees", "Celsius"],
        description="Basic scientific fact recall",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="The largest planet in our solar system is",
        category=PromptCategory.FACTUAL_RECALL,
        expected_tokens=["Jupiter"],
        description="Basic astronomy fact recall",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="The author of Romeo and Juliet is",
        category=PromptCategory.FACTUAL_RECALL,
        expected_tokens=["William", "Shakespeare"],
        description="Literary fact recall",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="The chemical symbol for gold is",
        category=PromptCategory.FACTUAL_RECALL,
        expected_tokens=["Au"],
        description="Chemistry fact recall",
        max_tokens=10,
    ),
]


# ============================================================================
# SIMPLE REASONING
# ============================================================================

SIMPLE_REASONING_PROMPTS = [
    EvaluationPrompt(
        prompt="If all cats are animals, and all animals are living things, then cats are",
        category=PromptCategory.SIMPLE_REASONING,
        expected_tokens=["living", "things"],
        description="Simple syllogism reasoning",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="John is taller than Mary. Mary is taller than Sue. Who is the tallest?",
        category=PromptCategory.SIMPLE_REASONING,
        expected_tokens=["John"],
        description="Transitive reasoning",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="A square has four sides. A triangle has three sides. Which has more sides?",
        category=PromptCategory.SIMPLE_REASONING,
        expected_tokens=["square"],
        description="Comparative reasoning",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="Complete the pattern: 2, 4, 6, 8,",
        category=PromptCategory.SIMPLE_REASONING,
        expected_tokens=["10"],
        description="Arithmetic sequence completion",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?",
        category=PromptCategory.SIMPLE_REASONING,
        expected_tokens=["not", "necessarily"],
        description="Logical fallacy recognition (affirming the consequent)",
        max_tokens=20,
    ),
]


# ============================================================================
# ARITHMETIC
# ============================================================================

ARITHMETIC_PROMPTS = [
    EvaluationPrompt(
        prompt="2 + 2 =",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["4"],
        description="Single digit addition",
        max_tokens=5,
    ),
    EvaluationPrompt(
        prompt="15 - 7 =",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["8"],
        description="Subtraction",
        max_tokens=5,
    ),
    EvaluationPrompt(
        prompt="6 * 7 =",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["42"],
        description="Multiplication",
        max_tokens=5,
    ),
    EvaluationPrompt(
        prompt="100 / 4 =",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["25"],
        description="Division",
        max_tokens=5,
    ),
    EvaluationPrompt(
        prompt="What is 12 times 12?",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["144"],
        description="Multiplication word problem",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="Calculate: (5 + 3) * 2 =",
        category=PromptCategory.ARITHMETIC,
        expected_tokens=["16"],
        description="Order of operations",
        max_tokens=5,
    ),
]


# ============================================================================
# CODE COMPLETION
# ============================================================================

CODE_COMPLETION_PROMPTS = [
    EvaluationPrompt(
        prompt="def add(a, b):\n    return",
        category=PromptCategory.CODE_COMPLETION,
        expected_tokens=["a + b"],
        description="Simple function completion",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="for i in range(10):\n    print(i)",
        category=PromptCategory.CODE_COMPLETION,
        expected_tokens=["print", "i"],
        description="Loop completion",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="class MyClass:\n    def __init__(self):\n        self.value =",
        category=PromptCategory.CODE_COMPLETION,
        expected_tokens=["0"],
        description="Class initialization completion",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="import json\n\ndata = {'key': 'value'}\njson_string = json.",
        category=PromptCategory.CODE_COMPLETION,
        expected_tokens=["dumps", "("],
        description="Library function completion",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(",
        category=PromptCategory.CODE_COMPLETION,
        expected_tokens=["n - 1"],
        description="Recursive function completion",
        max_tokens=20,
    ),
]


# ============================================================================
# INSTRUCTION FOLLOWING
# ============================================================================

INSTRUCTION_FOLLOWING_PROMPTS = [
    EvaluationPrompt(
        prompt="List three colors:",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        expected_tokens=["red", "blue", "green"],
        description="List generation following instruction",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="Write a short sentence using the word 'hello':",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        expected_tokens=["hello"],
        description="Constrained sentence generation",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="Count from 1 to 5:",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        expected_tokens=["1", "2", "3", "4", "5"],
        description="Sequential counting instruction",
        max_tokens=20,
    ),
    EvaluationPrompt(
        prompt="Reverse the word 'hello':",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        expected_tokens=["olleh"],
        description="String manipulation instruction",
        max_tokens=10,
    ),
    EvaluationPrompt(
        prompt="Answer with only 'yes' or 'no': Is water wet?",
        category=PromptCategory.INSTRUCTION_FOLLOWING,
        expected_tokens=["yes"],
        description="Constrained binary answer",
        max_tokens=5,
    ),
]


# ============================================================================
# ALL PROMPTS COMBINED
# ============================================================================

ALL_PROMPTS = (
    FACTUAL_RECALL_PROMPTS +
    SIMPLE_REASONING_PROMPTS +
    ARITHMETIC_PROMPTS +
    CODE_COMPLETION_PROMPTS +
    INSTRUCTION_FOLLOWING_PROMPTS
)


def get_prompts_by_category(category: PromptCategory) -> List[EvaluationPrompt]:
    """Get all prompts for a specific category."""
    mapping = {
        PromptCategory.FACTUAL_RECALL: FACTUAL_RECALL_PROMPTS,
        PromptCategory.SIMPLE_REASONING: SIMPLE_REASONING_PROMPTS,
        PromptCategory.ARITHMETIC: ARITHMETIC_PROMPTS,
        PromptCategory.CODE_COMPLETION: CODE_COMPLETION_PROMPTS,
        PromptCategory.INSTRUCTION_FOLLOWING: INSTRUCTION_FOLLOWING_PROMPTS,
    }
    return mapping.get(category, [])


def get_all_prompts() -> List[EvaluationPrompt]:
    """Get all evaluation prompts."""
    return list(ALL_PROMPTS)


def get_prompt_suite_summary() -> Dict[str, int]:
    """Get summary of prompt suite."""
    return {
        "factual_recall": len(FACTUAL_RECALL_PROMPTS),
        "simple_reasoning": len(SIMPLE_REASONING_PROMPTS),
        "arithmetic": len(ARITHMETIC_PROMPTS),
        "code_completion": len(CODE_COMPLETION_PROMPTS),
        "instruction_following": len(INSTRUCTION_FOLLOWING_PROMPTS),
        "total": len(ALL_PROMPTS),
    }


def print_prompt_suite() -> None:
    """Print all prompts with metadata."""
    print("=" * 80)
    print("LDMARK EVALUATION PROMPT SUITE")
    print("=" * 80)
    
    for category in PromptCategory:
        prompts = get_prompts_by_category(category)
        print(f"\n## {category.value.upper()} ({len(prompts)} prompts)")
        print("-" * 40)
        
        for i, p in enumerate(prompts):
            print(f"\n{i+1}. [{p.max_tokens} tokens, temp={p.temperature}] {p.description}")
            print(f"   Prompt: {p.prompt!r}")
            if p.expected_tokens:
                print(f"   Expected: {p.expected_tokens}")
            if p.expected_answer:
                print(f"   Answer: {p.expected_answer}")
    
    summary = get_prompt_suite_summary()
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {summary['total']} prompts across {len(summary)-1} categories")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print_prompt_suite()