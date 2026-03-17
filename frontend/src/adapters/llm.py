import json
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

from config import settings
from adapters.base import LLMAdapter, Message, PipelineContext
from adapters.mcp_client import MCPSubjectMatterClient

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


class LMStudioAdapter(LLMAdapter):
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.lm_studio_base_url,
            api_key="lm-studio",  # LM Studio ignores the key
        )
        self._mcp = MCPSubjectMatterClient()

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        """
        Convert to OpenAI format, folding any leading system message into the
        first user message. Gemma 3 (and some other models) require strictly
        alternating user/assistant roles with no system role.
        """
        result = []
        system_prefix = ""

        for msg in messages:
            if msg.role == "system":
                system_prefix = msg.content + "\n\n"
            elif msg.role == "user" and system_prefix:
                result.append({"role": "user", "content": system_prefix + msg.content})
                system_prefix = ""
            else:
                result.append({"role": msg.role, "content": msg.content})

        return result

    def _get_synthetic_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image for the user interface if a visual subject is discussed.",
                    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_music",
                    "description": "Generate a short music or audio clip if explicitly discussed.",
                    "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_function",
                    "description": "Plot a mathematical function using pure python math/matplotlib.",
                    "parameters": {"type": "object", "properties": {"python_code": {"type": "string"}}, "required": ["python_code"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": "Sound out a word or short phrase aloud when the student asks how to pronounce it.",
                    "parameters": {"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "record_interaction",
                    "description": (
                        "Rate the student's demonstrated understanding AFTER you have finished responding. "
                        "Call this once per turn when the student gave an answer or showed their thinking. "
                        "Skip for media-only requests (draw, plot, play music, show me X). "
                        "Rate the STUDENT's knowledge — not the topic difficulty."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Concept path, e.g. 'mathematics/gaussian_distribution' or 'music/intervals'"
                            },
                            "mastery_signal": {
                                "type": "integer",
                                "description": "0=confused, 1=misconception, 2=partial, 3=mostly correct, 4=fully mastered; -1 if student only asked a question"
                            },
                            "approach": {
                                "type": "string",
                                "enum": ["answered", "questioned", "struggled", "demonstrated"],
                                "description": "How the student engaged this turn"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Brief observation (≤100 chars), e.g. 'confuses σ with σ²'"
                            }
                        },
                        "required": ["topic", "mastery_signal", "approach"]
                    }
                }
            }
        ]

    async def generate(self, messages: list[Message], ctx: PipelineContext | None = None, _depth: int = 0) -> str:
        tools = await self._mcp.get_tools()
        tools.extend(self._get_synthetic_tools())

        response = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
            tools=tools,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(Message(role="assistant", content=msg.content or "", tool_calls=msg.model_dump().get("tool_calls", [])))

            has_mcp_call = False

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name == "generate_image":
                    if ctx: ctx.pending_image_prompt = fn_args.get("prompt")
                    result_text = "ok"
                elif fn_name == "generate_music":
                    if ctx: ctx.pending_music_prompt = fn_args.get("prompt")
                    result_text = "ok"
                elif fn_name == "plot_function":
                    if ctx: ctx.pending_plot_code = fn_args.get("python_code")
                    result_text = "ok"
                elif fn_name == "speak":
                    if ctx: ctx.pending_speak_text = fn_args.get("word")
                    result_text = "ok"
                elif fn_name == "record_interaction":
                    if ctx: ctx.pending_interaction = fn_args
                    result_text = "ok"
                else:
                    result_text = await self._mcp.call_tool(fn_name, fn_args)
                    has_mcp_call = True

                messages.append(Message(role="tool", content=result_text, tool_call_id=tool_call.id))

            # Recurse when:
            # (a) MCP tools returned data the LLM needs to synthesise, OR
            # (b) The model emitted only tool calls with no text — recurse to get the explanation.
            #     Empty synthetic results prevent an "Okay, I've done it" acknowledgement.
            should_recurse = has_mcp_call or not (msg.content or "").strip()
            if should_recurse and _depth < MAX_TOOL_ROUNDS:
                return await self.generate(messages, ctx, _depth + 1)
            if should_recurse:
                logger.warning("Tool-call depth limit (%d) reached — returning partial response", MAX_TOOL_ROUNDS)
            return msg.content or ""

        return msg.content or ""

    async def stream(self, messages: list[Message], ctx: PipelineContext | None = None, _depth: int = 0) -> AsyncIterator[str]:
        tools = await self._mcp.get_tools()
        tools.extend(self._get_synthetic_tools())

        stream_response = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
            stream=True,
            tools=tools,
        )

        tool_calls_buffer = {}
        text_yielded = False

        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.id or f"call_{idx}",
                                "name": (tc.function.name if tc.function else "") or "",
                                "arguments": (tc.function.arguments if tc.function else "") or ""
                            }
                        else:
                            if tc.function and tc.function.name:
                                tool_calls_buffer[idx]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls_buffer[idx]["arguments"] += tc.function.arguments
                elif delta.content and not tool_calls_buffer:
                    # Only yield normal content if we aren't currently capturing a tool call
                    yield delta.content
                    text_yielded = True

        if tool_calls_buffer:
            messages.append(Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                } for tc in tool_calls_buffer.values()]
            ))

            has_mcp_call = False

            for tc in tool_calls_buffer.values():
                fn_name = tc["name"]
                try:
                    fn_args = json.loads(tc["arguments"])
                except Exception:
                    fn_args = {}

                if fn_name == "generate_image":
                    if ctx: ctx.pending_image_prompt = fn_args.get("prompt")
                    result_text = "ok"
                elif fn_name == "generate_music":
                    if ctx: ctx.pending_music_prompt = fn_args.get("prompt")
                    result_text = "ok"
                elif fn_name == "plot_function":
                    if ctx: ctx.pending_plot_code = fn_args.get("python_code")
                    result_text = "ok"
                elif fn_name == "speak":
                    if ctx: ctx.pending_speak_text = fn_args.get("word")
                    result_text = "ok"
                elif fn_name == "record_interaction":
                    if ctx: ctx.pending_interaction = fn_args
                    result_text = "ok"
                else:
                    result_text = await self._mcp.call_tool(fn_name, fn_args)
                    has_mcp_call = True

                messages.append(Message(role="tool", content=result_text, tool_call_id=tc["id"]))

            # Recurse when:
            # (a) MCP tools returned data the LLM needs to synthesise, OR
            # (b) The model emitted only tool calls with no text — recurse to get the explanation.
            #     Empty synthetic results prevent an "Okay, I've done it" acknowledgement.
            should_recurse = has_mcp_call or not text_yielded
            if should_recurse and _depth < MAX_TOOL_ROUNDS:
                async for next_chunk in self.stream(messages, ctx, _depth + 1):
                    yield next_chunk
            elif should_recurse:
                logger.warning("Tool-call depth limit (%d) reached in stream — returning partial response", MAX_TOOL_ROUNDS)
