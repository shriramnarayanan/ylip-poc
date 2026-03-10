"""
Pipeline engine for chaining and parallelising model adapter calls.

A pipeline is a list of Steps. Each Step is either:
  - Step: a single async function, receives the context, returns a partial context update
  - ParallelStep: a group of Steps that run concurrently via asyncio.gather

Example — creative pipeline (LLM first, then TTS + image gen in parallel):

    pipeline = [
        Step("llm",      llm_step_fn),
        ParallelStep([
            Step("tts",       tts_step_fn),
            Step("image_gen", image_gen_step_fn),
        ]),
    ]
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

from adapters.base import PipelineContext

StepFn = Callable[[PipelineContext], Awaitable[PipelineContext]]


@dataclass
class Step:
    name: str
    fn: StepFn


@dataclass
class ParallelStep:
    """Runs all contained steps concurrently; merges their results into the context."""
    steps: list[Step]


async def run_pipeline(
    steps: list[Step | ParallelStep],
    context: PipelineContext,
) -> PipelineContext:
    for step in steps:
        if isinstance(step, ParallelStep):
            # Run all steps in the group concurrently.
            # Each returns a context; merge non-None output fields back.
            results = await asyncio.gather(*(s.fn(context) for s in step.steps))
            for result in results:
                _merge(context, result)
        else:
            context = await step.fn(context)
    return context


def _merge(target: PipelineContext, source: PipelineContext) -> None:
    """Copy non-None output fields from source into target."""
    for field in ("llm_response", "tts_audio", "generated_image", "vision_description"):
        value = getattr(source, field)
        if value is not None:
            setattr(target, field, value)
