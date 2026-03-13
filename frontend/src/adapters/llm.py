import json
from typing import AsyncIterator

from openai import AsyncOpenAI

from config import settings
from adapters.base import LLMAdapter, Message, PipelineContext
from adapters.mcp_client import MCPSubjectMatterClient


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
            }
        ]

    async def generate(self, messages: list[Message], ctx: PipelineContext | None = None) -> str:
        tools = await self._mcp.get_tools()
        tools.extend(self._get_synthetic_tools())
        
        kwargs = {"tools": tools}

        response = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
            **kwargs,
        )
        
        msg = response.choices[0].message
        
        # Tool execution loop
        if msg.tool_calls:
            # We append the original assistant response carrying the tool_calls
            messages.append(Message(role="assistant", content=msg.content or "", tool_calls=msg.model_dump().get("tool_calls", [])))
            
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                
                # Intercept Synthetic UI Side-Effects
                if fn_name == "generate_image":
                    if ctx: ctx.pending_image_prompt = fn_args.get("prompt")
                    result_text = "Image queued for UI rendering."
                elif fn_name == "generate_music":
                    if ctx: ctx.pending_music_prompt = fn_args.get("prompt")
                    result_text = "Music queued for UI rendering."
                elif fn_name == "plot_function":
                    if ctx: ctx.pending_plot_code = fn_args.get("python_code")
                    result_text = "Plot queued for UI rendering."
                elif fn_name == "speak":
                    if ctx: ctx.pending_speak_text = fn_args.get("word")
                    result_text = "Word queued for pronunciation."
                else:
                    # Execute remote MCP Subject Matter Tools
                    result_text = await self._mcp.call_tool(fn_name, fn_args)
                
                messages.append(Message(
                    role="tool", 
                    content=result_text,
                    tool_call_id=tool_call.id
                ))
            
            # Recurse with the newly appended tool context
            return await self.generate(messages, ctx)

        return msg.content or ""

    async def stream(self, messages: list[Message], ctx: PipelineContext | None = None) -> AsyncIterator[str]:
        tools = await self._mcp.get_tools()
        tools.extend(self._get_synthetic_tools())
        kwargs = {"tools": tools}

        stream_response = await self._client.chat.completions.create(
            model=settings.lm_studio_model,
            messages=self._to_openai(messages),
            max_tokens=settings.lm_studio_max_tokens,
            temperature=settings.lm_studio_temperature,
            stream=True,
            **kwargs
        )
        
        tool_calls_buffer = {}

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

            for tc in tool_calls_buffer.values():
                fn_name = tc["name"]
                try:
                    fn_args = json.loads(tc["arguments"])
                except Exception:
                    fn_args = {}
                
                # Intercept Synthetic UI Side-Effects
                if fn_name == "generate_image":
                    if ctx: ctx.pending_image_prompt = fn_args.get("prompt")
                    result_text = "Image queued for UI rendering."
                elif fn_name == "generate_music":
                    if ctx: ctx.pending_music_prompt = fn_args.get("prompt")
                    result_text = "Music queued for UI rendering."
                elif fn_name == "plot_function":
                    if ctx: ctx.pending_plot_code = fn_args.get("python_code")
                    result_text = "Plot queued for UI rendering."
                elif fn_name == "speak":
                    if ctx: ctx.pending_speak_text = fn_args.get("word")
                    result_text = "Word queued for pronunciation."
                else:
                    # Execute remote MCP Subject Matter Tools
                    result_text = await self._mcp.call_tool(fn_name, fn_args)
                
                messages.append(Message(
                    role="tool", 
                    content=str(result_text),
                    tool_call_id=tc["id"]
                ))
            
            # Recurse the stream with the populated tool contexts
            async for next_chunk in self.stream(messages, ctx):
                yield next_chunk
