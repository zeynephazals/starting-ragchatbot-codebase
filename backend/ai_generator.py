from anthropic import AnthropicFoundry
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Claude via Azure Anthropic Foundry"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content or detailed educational topics.
- **get_course_outline**: Retrieve a course's outline — its title, course link, and the full list of lessons (each lesson's number and title).

Tool Usage:
- Use **search_course_content** **only** for questions about specific course content or detailed educational materials.
- Use **get_course_outline** for questions about a course's structure, syllabus, or which lessons it contains.
- You may make **up to 2 tool calls in sequence** for a single query. Make a second call only after seeing the first result, when it supplies information needed for the next step — for example, calling **get_course_outline** to find a lesson's title, then **search_course_content** with that title to find related material across courses.
- Use a second call for comparisons, multi-part questions, or when one result tells you what to search for next.
- If one tool call fully answers the question, do not make a second.
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Course Outline Responses:
- When answering an outline query, return the course title, the course link, and the complete lesson list.
- For each lesson, include its lesson number and lesson title.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    # Maximum number of sequential tool-use rounds per user query.
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        self.client = AnthropicFoundry(api_key=api_key, base_url=base_url)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool-use rounds: each round is a
        separate API request, so Claude can reason about previous tool results
        before deciding whether to call another tool. The loop terminates when
        (a) MAX_TOOL_ROUNDS rounds have completed, (b) Claude returns no tool_use
        blocks, or (c) tools are withheld on the final round to force a text answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Conversation context accumulates here across rounds.
        messages = [{"role": "user", "content": query}]

        # Without a tool_manager we can't run a tool loop. Still offer the tools so
        # Claude can see them (preserves the original single-call contract), but make
        # just one call. With no tools at all this is a plain completion.
        if not (tools and tool_manager):
            return self._extract_text(self._call_api(messages, system_content, tools=tools))

        # Sequential tool-use loop: each iteration is one tool round. Tools are
        # offered every round so Claude can chain up to MAX_TOOL_ROUNDS calls,
        # reasoning over each result before the next.
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self._call_api(messages, system_content, tools=tools)

            # (b) Claude answered without requesting a tool — return its text.
            if response.stop_reason != "tool_use":
                return self._extract_text(response)

            # Record the assistant's tool_use turn, then execute the tools and feed
            # the results back so the next round can reason over them.
            messages.append({"role": "assistant", "content": response.content})
            tool_results = self._run_tools(response, tool_manager)
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # (a) Round cap reached and Claude still wanted a tool. Make one final call
        # WITHOUT tools to force a synthesized text answer.
        return self._extract_text(self._call_api(messages, system_content))

    def _call_api(self, messages: List[Dict[str, Any]], system_content: str,
                  tools: Optional[List] = None):
        """Make a single Claude API call, optionally offering tools."""
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**api_params)

    def _run_tools(self, response, tool_manager) -> List[Dict[str, Any]]:
        """
        Execute every tool_use block in a response and return tool_result blocks.

        A tool that raises is caught and surfaced to Claude as an error
        tool_result, so the model can react gracefully rather than crashing the
        request. (Tools that return error *strings* flow through normally.)
        """
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                content = tool_manager.execute_tool(block.name, **block.input)
            except Exception as e:
                content = f"Tool '{block.name}' failed: {e}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        return tool_results

    def _extract_text(self, response) -> str:
        """Return the first text block's text, or "" if the response has none."""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""