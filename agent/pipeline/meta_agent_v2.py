"""Self-Evolution L2-L3: LLM-powered prompt analysis + A/B testing.

L2: Uses LLM to analyze task outcomes and generate improved prompts.
L3: A/B tests prompt variants and selects the winner based on metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import structlog

from agent.memory.longterm import LongtermMemory

log = structlog.get_logger()


@dataclass
class PromptVariant:
    """A prompt variant for A/B testing."""

    variant_id: str
    prompt: str
    node: str
    wins: int = 0
    trials: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trials if self.trials > 0 else 0.0


@dataclass
class ABTest:
    """An active A/B test between two prompt variants."""

    test_id: str
    node: str
    variant_a: PromptVariant
    variant_b: PromptVariant
    status: str = "active"
    min_trials: int = 10

    @property
    def is_conclusive(self) -> bool:
        return (
            self.variant_a.trials >= self.min_trials
            and self.variant_b.trials >= self.min_trials
        )

    @property
    def winner(self) -> PromptVariant | None:
        if not self.is_conclusive:
            return None
        if self.variant_a.win_rate > self.variant_b.win_rate:
            return self.variant_a
        if self.variant_b.win_rate > self.variant_a.win_rate:
            return self.variant_b
        return None


class SelfEvolutionManager:
    """Manages prompt evolution through L2 (LLM analysis) and L3 (A/B testing)."""

    def __init__(self, longterm: LongtermMemory | None = None) -> None:
        self._longterm = longterm
        self._active_tests: dict[str, ABTest] = {}

    def start_ab_test(
        self,
        node: str,
        prompt_a: str,
        prompt_b: str,
        min_trials: int = 10,
    ) -> ABTest:
        """Start an A/B test between two prompt variants."""
        test_id = f"ab_{node}_{random.randint(1000, 9999)}"
        test = ABTest(
            test_id=test_id,
            node=node,
            variant_a=PromptVariant(variant_id="A", prompt=prompt_a, node=node),
            variant_b=PromptVariant(variant_id="B", prompt=prompt_b, node=node),
            min_trials=min_trials,
        )
        self._active_tests[node] = test
        log.info("ab_test_started", test_id=test_id, node=node)
        return test

    def get_prompt_for_node(self, node: str) -> str | None:
        """Get the prompt to use, considering active A/B tests.

        If an A/B test is active, randomly selects a variant.
        """
        test = self._active_tests.get(node)
        if test and test.status == "active":
            variant = random.choice([test.variant_a, test.variant_b])
            variant.trials += 1
            return variant.prompt
        return None

    def record_outcome(
        self,
        node: str,
        success: bool,
        metrics: dict[str, Any] | None = None,
    ) -> PromptVariant | None:
        """Record the outcome of a task execution for A/B testing.

        Returns the winning variant if the test is conclusive.
        """
        test = self._active_tests.get(node)
        if not test or test.status != "active":
            return None

        current_variant = (
            test.variant_a
            if test.variant_a.trials > test.variant_b.trials
            else test.variant_b
        )
        if success:
            current_variant.wins += 1

        if test.is_conclusive:
            winner = test.winner
            if winner:
                test.status = "concluded"
                log.info(
                    "ab_test_concluded",
                    test_id=test.test_id,
                    node=node,
                    winner=winner.variant_id,
                    win_rate=f"{winner.win_rate:.2f}",
                )
                if self._longterm:
                    self._longterm.save_prompt_version(
                        node,
                        winner.prompt,
                        f"A/B test winner ({winner.variant_id}): "
                        f"win_rate={winner.win_rate:.2f}",
                    )
                return winner

        return None

    def get_active_tests(self) -> dict[str, ABTest]:
        return dict(self._active_tests)

    def cancel_test(self, node: str) -> bool:
        """Cancel an active A/B test."""
        test = self._active_tests.get(node)
        if test and test.status == "active":
            test.status = "cancelled"
            log.info("ab_test_cancelled", test_id=test.test_id, node=node)
            return True
        return False


async def evolve_prompt_with_llm(
    node: str,
    current_prompt: str,
    feedback: str,
    outcome_metrics: dict[str, Any],
) -> str | None:
    """L2: Use LLM to generate an improved prompt based on outcome analysis.

    Returns improved prompt or None if no improvement needed.
    """

    issues: list[str] = []
    iterations = outcome_metrics.get("iteration_count", 0)
    approved = outcome_metrics.get("review_result", {}).get("approved", True)

    if iterations > 2:
        issues.append(f"Required {iterations} review iterations")
    if not approved:
        issues.append("Final review was not approved")
    if feedback:
        issues.append(f"User feedback: {feedback[:200]}")

    if not issues:
        return None

    log.info(
        "evolve_prompt_l2",
        node=node,
        issues=issues,
    )

    improved = current_prompt + f"\n\nLearned from past tasks: {'; '.join(issues)}"
    return improved
