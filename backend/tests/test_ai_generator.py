"""Tests for AIGenerator (backend/ai_generator.py).

The Anthropic client is mocked — we are testing whether AIGenerator drives the
tool-use loop correctly: passing tools to the API, dispatching tool_use blocks
to the ToolManager, and returning the model's final text.
"""

from unittest.mock import MagicMock, call, patch

from tests.conftest import make_response, make_text_block, make_tool_use_block


def _make_generator():
    """Construct an AIGenerator with a mocked Anthropic client."""
    with patch("ai_generator.AnthropicFoundry"):
        from ai_generator import AIGenerator

        gen = AIGenerator(api_key="k", model="claude-haiku-4-5", base_url="http://x")
        # gen.client is the mocked instance
        return gen, gen.client


def test_no_tool_use_returns_text():
    gen, client = _make_generator()
    client.messages.create.return_value = make_response(
        "end_turn", [make_text_block("Paris is the capital of France.")]
    )

    out = gen.generate_response(query="capital of France?")

    assert out == "Paris is the capital of France."
    assert client.messages.create.call_count == 1


def test_tools_are_passed_to_api():
    gen, client = _make_generator()
    client.messages.create.return_value = make_response(
        "end_turn", [make_text_block("hi")]
    )
    tools = [{"name": "search_course_content"}]

    gen.generate_response(query="q", tools=tools)

    sent = client.messages.create.call_args.kwargs
    assert sent["tools"] == tools
    assert sent["tool_choice"] == {"type": "auto"}


def test_tool_use_dispatches_to_tool_manager():
    gen, client = _make_generator()

    # First call: model asks to use the search tool. Second call: final answer.
    first = make_response(
        "tool_use",
        [
            make_tool_use_block(
                "search_course_content", {"query": "what is mcp"}, "tu_1"
            )
        ],
    )
    second = make_response("end_turn", [make_text_block("MCP is a protocol.")])
    client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = (
        "[MCP Basics - Lesson 1]\nMCP is a protocol."
    )

    tools = [{"name": "search_course_content"}]
    out = gen.generate_response(
        query="what is mcp", tools=tools, tool_manager=tool_manager
    )

    # The search tool must be invoked with the model-provided input.
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="what is mcp"
    )
    # Final synthesized answer is returned.
    assert out == "MCP is a protocol."
    assert client.messages.create.call_count == 2


def test_tool_result_is_fed_back_to_model():
    gen, client = _make_generator()
    first = make_response(
        "tool_use",
        [make_tool_use_block("search_course_content", {"query": "q"}, "tu_9")],
    )
    second = make_response("end_turn", [make_text_block("done")])
    client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "SEARCH_PAYLOAD"

    gen.generate_response(
        query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    # Inspect the second (final) API call's messages: it must contain the
    # tool_result carrying our search payload, keyed to the tool_use id.
    final_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_results = [
        block
        for msg in final_messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "tu_9"
    assert tool_results[0]["content"] == "SEARCH_PAYLOAD"


def _tool_results_in(messages):
    """Collect all tool_result blocks across a messages list."""
    return [
        block
        for msg in messages
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
        if isinstance(block, dict) and block.get("type") == "tool_result"
    ]


def test_two_round_sequential_flow():
    """Claude calls get_course_outline, reasons over the result, then searches."""
    gen, client = _make_generator()

    first = make_response(
        "tool_use",
        [make_tool_use_block("get_course_outline", {"course_name": "X"}, "t1")],
    )
    second = make_response(
        "tool_use",
        [
            make_tool_use_block(
                "search_course_content", {"query": "Lesson 4 Topic"}, "t2"
            )
        ],
    )
    third = make_response("end_turn", [make_text_block("Course Y covers it.")])
    client.messages.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = [
        "Course X\nLesson 4: Lesson 4 Topic",
        "[Course Y - Lesson 2]\nSame topic.",
    ]

    tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
    out = gen.generate_response(
        query="find a course like lesson 4 of X", tools=tools, tool_manager=tool_manager
    )

    # Three API calls: outline round, search round, final synthesis.
    assert client.messages.create.call_count == 3
    # Both tools executed in order with the model-provided inputs.
    assert tool_manager.execute_tool.call_args_list == [
        call("get_course_outline", course_name="X"),
        call("search_course_content", query="Lesson 4 Topic"),
    ]
    assert out == "Course Y covers it."


def test_max_rounds_caps_tool_calls_then_forces_text_answer():
    """If Claude keeps requesting tools, it's capped at MAX_TOOL_ROUNDS calls and
    a final tools-less call forces a text answer."""
    gen, client = _make_generator()

    # Claude requests a tool on both allowed rounds (greedy)...
    first = make_response(
        "tool_use", [make_tool_use_block("search_course_content", {"query": "a"}, "t1")]
    )
    second = make_response(
        "tool_use", [make_tool_use_block("search_course_content", {"query": "b"}, "t2")]
    )
    # ...and the forced final (tools-less) call yields the text answer.
    third = make_response("end_turn", [make_text_block("answer")])
    client.messages.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "PAYLOAD"

    out = gen.generate_response(
        query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    assert out == "answer"
    # Two tool rounds + one synthesis call = 3 API calls.
    assert client.messages.create.call_count == 3
    # Tools executed exactly twice — capped at MAX_TOOL_ROUNDS.
    assert tool_manager.execute_tool.call_count == 2
    # The final synthesis call must NOT offer tools.
    final_kwargs = client.messages.create.call_args_list[2].kwargs
    assert "tools" not in final_kwargs


def test_tool_exception_is_fed_back_to_model():
    """A tool that raises is surfaced to Claude as an error tool_result."""
    gen, client = _make_generator()

    first = make_response(
        "tool_use",
        [make_tool_use_block("search_course_content", {"query": "q"}, "t1")],
    )
    second = make_response("end_turn", [make_text_block("Sorry, the search failed.")])
    client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("boom")

    out = gen.generate_response(
        query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
    )

    # No exception escaped; Claude got a chance to respond.
    assert out == "Sorry, the search failed."
    # The second call carries an error tool_result describing the failure.
    final_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_results = _tool_results_in(final_messages)
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "t1"
    assert "boom" in tool_results[0]["content"]


def test_tool_error_string_flows_back_normally():
    """A tool returning an error *string* is not a failure — it feeds back as-is."""
    gen, client = _make_generator()

    first = make_response(
        "tool_use",
        [make_tool_use_block("get_course_outline", {"course_name": "Z"}, "t1")],
    )
    second = make_response("end_turn", [make_text_block("No such course.")])
    client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "No course found matching 'Z'."

    out = gen.generate_response(
        query="outline of Z",
        tools=[{"name": "get_course_outline"}],
        tool_manager=tool_manager,
    )

    assert out == "No such course."
    final_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    tool_results = _tool_results_in(final_messages)
    assert tool_results[0]["content"] == "No course found matching 'Z'."


def test_final_text_extracted_when_text_block_not_first():
    """_extract_text scans past a leading non-text block instead of indexing [0]."""
    gen, client = _make_generator()
    client.messages.create.return_value = make_response(
        "end_turn",
        [
            make_tool_use_block("search_course_content", {"query": "q"}, "t1"),
            make_text_block("the real answer"),
        ],
    )

    out = gen.generate_response(query="q")

    assert out == "the real answer"
