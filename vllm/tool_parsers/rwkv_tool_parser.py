# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

_TOOL_CALL_MARKER = "**Tool Call:**"
_LEGACY_TOOL_CALL_MARKER = "<tool_call>"
_JSON_FENCE_START = "```json"
_FENCED_JSON_RE = re.compile(
    r"\*\*Tool Call:\*\*\s*```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)
_STANDALONE_FENCED_JSON_RE = re.compile(
    r"```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)
_CANONICAL_TOOL_START_RE = re.compile(
    r'\{\s*"(?P<kind>name|tool_calls)"\s*:',
)
_CANONICAL_NAME_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>(?:\\.|[^"\\])*)"',
    re.DOTALL,
)
_CANONICAL_ARGUMENTS_KEY_RE = re.compile(r'"arguments"\s*:\s*')


@dataclass(frozen=True)
class _ToolCallMatch:
    start: int
    payload: dict[str, Any]


class RWKVToolParser(ToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0
        self._canonical_stream_id: str | None = None
        self._canonical_stream_start = -1
        self._canonical_stream_args_sent = 0
        self._canonical_stream_name_sent = False

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)

    def _allowed_tool_names(self) -> set[str]:
        names = set[str]()
        for tool in self.tools:
            function = getattr(tool, "function", tool)
            name = getattr(function, "name", None)
            if name is not None:
                names.add(name)
        return names

    @staticmethod
    def _loads_mapping(payload: str) -> dict[str, Any] | None:
        parsed = RWKVToolParser._loads_value(payload)
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _loads_value(payload: str) -> Any:
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return None

    @staticmethod
    def _normalize_legacy_arguments(
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if "path" not in arguments and isinstance(arguments.get("filePath"), str):
            arguments = {**arguments, "path": arguments["filePath"]}

        if name == "read" and "path" in arguments:
            read_arguments = {"path": arguments["path"]}
            if "offset" in arguments:
                read_arguments["offset"] = arguments["offset"]
            if "limit" in arguments:
                read_arguments["limit"] = arguments["limit"]
            return read_arguments

        if name == "edit":
            path = arguments.get("path")
            edits = arguments.get("edits")
            if not isinstance(path, str) and isinstance(edits, list):
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    nested_path = edit.get("path", edit.get("filePath"))
                    if isinstance(nested_path, str):
                        path = nested_path
                        break

            if not isinstance(path, str):
                return arguments

            edit_arguments: dict[str, Any] = {"path": path}
            if isinstance(edits, list):
                normalized_edits = []
                for edit in edits:
                    if not isinstance(edit, dict):
                        continue
                    if {
                        "oldText",
                        "newText",
                    } <= edit.keys():
                        normalized_edits.append(
                            {
                                "oldText": edit["oldText"],
                                "newText": edit["newText"],
                            }
                        )
                if normalized_edits:
                    edit_arguments["edits"] = normalized_edits
            elif {
                "oldText",
                "newText",
            } <= arguments.keys():
                edit_arguments["edits"] = [
                    {
                        "oldText": arguments["oldText"],
                        "newText": arguments["newText"],
                    }
                ]
            return edit_arguments

        return arguments

    def _parse_tool_call_mapping(self, parsed: Any) -> dict[str, Any] | None:
        if not isinstance(parsed, dict):
            return None
        if "name" not in parsed and isinstance(parsed.get("function"), dict):
            function = parsed["function"]
            parsed = {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        if not isinstance(parsed.get("name"), str):
            return None
        if "arguments" not in parsed:
            return None
        if isinstance(parsed["arguments"], str):
            arguments = self._loads_mapping(parsed["arguments"])
            if arguments is None:
                return None
            parsed["arguments"] = arguments
        if not isinstance(parsed["arguments"], dict):
            return None
        parsed["arguments"] = self._normalize_legacy_arguments(
            parsed["name"],
            parsed["arguments"],
        )
        allowed_names = self._allowed_tool_names()
        if allowed_names and parsed["name"] not in allowed_names:
            return None
        return parsed

    def _parse_tool_call_value(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict) and "tool_calls" in value:
            value = value["tool_calls"]

        if isinstance(value, list):
            parsed_calls = []
            for item in value:
                parsed = self._parse_tool_call_mapping(item)
                if parsed is not None:
                    parsed_calls.append(parsed)
            return parsed_calls

        parsed = self._parse_tool_call_mapping(value)
        return [parsed] if parsed is not None else []

    def _parse_tool_call_payloads(self, payload: str) -> list[dict[str, Any]]:
        return self._parse_tool_call_value(self._loads_value(payload))

    def _iter_bare_json_matches(
        self,
        text: str,
        forbidden_ranges: Sequence[tuple[int, int]],
    ) -> list[_ToolCallMatch]:
        if not self.tools:
            return []

        decoder = json.JSONDecoder()
        matches: list[_ToolCallMatch] = []
        cursor = 0
        while cursor < len(text):
            starts = [text.find("{", cursor), text.find("[", cursor)]
            starts = [start for start in starts if start != -1]
            if not starts:
                break
            start = min(starts)
            forbidden_end = next(
                (
                    end
                    for range_start, end in forbidden_ranges
                    if range_start <= start < end
                ),
                None,
            )
            if forbidden_end is not None:
                cursor = forbidden_end
                continue

            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                cursor = start + 1
                continue

            payloads = self._parse_tool_call_value(value)
            matches.extend(_ToolCallMatch(start, payload) for payload in payloads)
            cursor = end if payloads else start + 1

        return matches

    def _iter_tool_call_matches(
        self,
        text: str,
    ) -> Sequence[_ToolCallMatch]:
        matches: list[_ToolCallMatch] = []
        marker_ranges: list[tuple[int, int]] = []
        fenced_ranges: list[tuple[int, int]] = []

        for match in _FENCED_JSON_RE.finditer(text):
            marker_ranges.append((match.start(), match.end()))
            matches.extend(
                _ToolCallMatch(match.start(), payload)
                for payload in self._parse_tool_call_payloads(match.group("json"))
            )

        marker_end = 0
        while (marker_start := text.find(_LEGACY_TOOL_CALL_MARKER, marker_end)) != -1:
            payload_start = marker_start + len(_LEGACY_TOOL_CALL_MARKER)
            line_start = text.find("{", payload_start)
            if line_start == -1:
                break
            line_end = text.find("\n", line_start)
            if line_end == -1:
                break
            matches.extend(
                _ToolCallMatch(marker_start, payload)
                for payload in self._parse_tool_call_payloads(
                    text[line_start:line_end].strip()
                )
            )
            marker_end = line_end

        for match in _STANDALONE_FENCED_JSON_RE.finditer(text):
            if any(start <= match.start() < end for start, end in marker_ranges):
                continue
            fenced_ranges.append((match.start(), match.end()))
            matches.extend(
                _ToolCallMatch(match.start(), payload)
                for payload in self._parse_tool_call_payloads(match.group("json"))
            )

        matches.extend(
            self._iter_bare_json_matches(text, marker_ranges + fenced_ranges)
        )

        unique_matches: dict[tuple[int, str, str], _ToolCallMatch] = {}
        for match in matches:
            key = (
                match.start,
                match.payload["name"],
                json.dumps(match.payload["arguments"], sort_keys=True),
            )
            unique_matches[key] = match
        return sorted(unique_matches.values(), key=lambda match: match.start)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        del request
        matches = self._iter_tool_call_matches(model_output)
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls: list[ToolCall] = []
        for match in matches:
            payload = match.payload
            tool_calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(
                        name=payload["name"],
                        arguments=self._serialize_arguments(payload["arguments"]),
                    ),
                )
            )

        content_end = matches[0].start
        canonical_match = _CANONICAL_TOOL_START_RE.search(model_output)
        if (
            canonical_match is not None
            and canonical_match.group("kind") == "tool_calls"
            and canonical_match.start() < content_end
        ):
            # If generation stops before the outer parallel-call envelope is
            # closed, the inner function object can still be parseable.  The
            # envelope is protocol text too, so it must not leak into content.
            content_end = canonical_match.start()
        content = model_output[:content_end]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _extract_content_delta(self, current_text: str) -> str | None:
        marker_starts = [
            current_text.find(_TOOL_CALL_MARKER),
            current_text.find(_LEGACY_TOOL_CALL_MARKER),
        ]
        if self.tools:
            marker_starts.append(current_text.find(_JSON_FENCE_START))
            canonical_match = _CANONICAL_TOOL_START_RE.search(current_text)
            if canonical_match is not None:
                marker_starts.append(canonical_match.start())
        marker_start = min(
            (start for start in marker_starts if start != -1),
            default=-1,
        )
        if marker_start != -1 and current_text[:marker_start].strip() == "":
            # Keep a whitespace-only prefix buffered until a tool marker is
            # confirmed. RWKV commonly emits one newline before JSON, which
            # should not become visible assistant content.
            self._sent_content_idx = marker_start
            return None
        if marker_start == -1:
            overlap = partial_tag_overlap(current_text, _TOOL_CALL_MARKER)
            overlap = max(
                overlap,
                partial_tag_overlap(current_text, _LEGACY_TOOL_CALL_MARKER),
            )
            if self.tools:
                overlap = max(
                    overlap,
                    partial_tag_overlap(current_text, _JSON_FENCE_START),
                    partial_tag_overlap(current_text, '{"name"'),
                    partial_tag_overlap(current_text, '{"tool_calls"'),
                )
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = marker_start

        if marker_start == -1 and (
            current_text[self._sent_content_idx : sendable_idx].strip() == ""
        ):
            return None

        if sendable_idx <= self._sent_content_idx:
            return None
        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, delta_text, previous_token_ids, current_token_ids
        del delta_token_ids, request

        content = self._extract_content_delta(current_text)
        tool_deltas: list[DeltaToolCall] = []

        canonical_delta = self._extract_canonical_tool_call_delta(current_text)
        if canonical_delta is not None:
            tool_deltas.append(canonical_delta)

        for index, match in enumerate(self._iter_tool_call_matches(current_text)):
            if index < len(self.prev_tool_call_arr):
                continue

            payload = match.payload
            arguments = self._serialize_arguments(payload["arguments"])
            self.prev_tool_call_arr.append(
                {
                    "name": payload["name"],
                    "arguments": arguments,
                }
            )
            self.streamed_args_for_tool.append(arguments)
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=payload["name"],
                        arguments=arguments,
                    ),
                )
            )

        if content or tool_deltas:
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        return None

    def _extract_canonical_tool_call_delta(
        self, current_text: str
    ) -> DeltaToolCall | None:
        """Stream a single bare JSON call without waiting for its closing brace."""
        if not self.tools:
            return None
        start_match = _CANONICAL_TOOL_START_RE.search(current_text)
        if start_match is None or start_match.group("kind") != "name":
            return None

        start = start_match.start()
        if any(
            match.start() <= start < match.end()
            for match in _STANDALONE_FENCED_JSON_RE.finditer(current_text)
        ) or self._is_inside_json_fence(current_text, start):
            return None
        name_match = _CANONICAL_NAME_RE.search(current_text, start)
        arguments_match = _CANONICAL_ARGUMENTS_KEY_RE.search(
            current_text, name_match.end() if name_match else start
        )
        if name_match is None or arguments_match is None:
            return None

        try:
            name = json.loads(f'"{name_match.group("name")}"')
        except json.JSONDecodeError:
            return None
        if not isinstance(name, str) or name not in self._allowed_tool_names():
            return None

        if self._canonical_stream_start != start:
            self._canonical_stream_start = start
            self._canonical_stream_id = make_tool_call_id()
            self._canonical_stream_args_sent = 0
            self._canonical_stream_name_sent = False
            self.prev_tool_call_arr.append({"name": name, "arguments": ""})
            self.streamed_args_for_tool.append("")

        argument_start = arguments_match.end()
        argument_end, _complete = self._scan_json_value(current_text, argument_start)
        visible_arguments = current_text[argument_start:argument_end]
        if len(visible_arguments) <= self._canonical_stream_args_sent:
            if self._canonical_stream_name_sent:
                return None
            self._canonical_stream_name_sent = True
            return DeltaToolCall(
                index=len(self.prev_tool_call_arr) - 1,
                type="function",
                id=self._canonical_stream_id,
                function=DeltaFunctionCall(name=name, arguments=""),
            )

        is_first_delta = self._canonical_stream_args_sent == 0
        self._canonical_stream_name_sent = True
        argument_delta = visible_arguments[self._canonical_stream_args_sent :]
        self._canonical_stream_args_sent = len(visible_arguments)
        index = len(self.prev_tool_call_arr) - 1
        self.streamed_args_for_tool[index] += argument_delta
        self.prev_tool_call_arr[index]["arguments"] = self.streamed_args_for_tool[index]
        return DeltaToolCall(
            index=index,
            type="function",
            id=self._canonical_stream_id if is_first_delta else None,
            function=DeltaFunctionCall(
                name=name if is_first_delta else None,
                arguments=argument_delta,
            ),
        )

    @staticmethod
    def _is_inside_json_fence(text: str, start: int) -> bool:
        opening = text.rfind(_JSON_FENCE_START, 0, start + 1)
        closing = (
            text.rfind("```", opening + len("```"), start) if opening != -1 else -1
        )
        return opening != -1 and closing < opening

    @staticmethod
    def _scan_json_value(text: str, start: int) -> tuple[int, bool]:
        if start >= len(text):
            return start, False
        if text[start] not in "[{":
            return start, False

        stack: list[str] = []
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack or (stack[-1], char) not in (("[", "]"), ("{", "}")):
                    return index, False
                stack.pop()
                if not stack:
                    return index + 1, True
        return len(text), False

    @staticmethod
    def _serialize_arguments(arguments: dict[str, Any]) -> str:
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
