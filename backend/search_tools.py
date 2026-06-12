from abc import ABC, abstractmethod
from typing import Any

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata, strict=False):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build the visible source label
            label = course_title
            if lesson_num is not None:
                label += f" - Lesson {lesson_num}"

            # Look up the lesson link and embed it invisibly as a clickable
            # anchor (only the label is shown; the URL stays hidden)
            link = None
            if lesson_num is not None:
                link = self.store.get_lesson_link(course_title, lesson_num)

            if link:
                sources.append(
                    f'<a href="{link}" target="_blank" rel="noopener noreferrer">{label}</a>'
                )
            else:
                sources.append(label)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for returning a course outline: title, link, and full lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last lookup

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the outline of a course: its title, link, and the full list of lessons (number and title for each). Use this for questions about a course's structure, syllabus, or what lessons it contains.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Execute the outline lookup.

        Args:
            course_name: Course title (partial matches supported)

        Returns:
            Formatted course outline or an error message
        """
        # Resolve the (possibly partial) course name to a full title
        course_title = self.store._resolve_course_name(course_name)
        if not course_title:
            return f"No course found matching '{course_name}'."

        # Fetch the course metadata from the catalog collection
        try:
            results = self.store.course_catalog.get(ids=[course_title])
        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"

        if not results or not results.get("metadatas"):
            return f"No metadata found for course '{course_title}'."

        metadata = results["metadatas"][0]
        return self._format_outline(metadata)

    def _format_outline(self, metadata: dict[str, Any]) -> str:
        """Format the course outline from catalog metadata"""
        import json

        title = metadata.get("title", "unknown")
        course_link = metadata.get("course_link")

        lines = [f"Course: {title}"]
        if course_link:
            lines.append(f"Course Link: {course_link}")

        lessons_json = metadata.get("lessons_json")
        lessons = json.loads(lessons_json) if lessons_json else []

        lines.append(f"Lessons ({len(lessons)}):")
        for lesson in sorted(lessons, key=lambda item: item.get("lesson_number", 0)):
            number = lesson.get("lesson_number")
            lesson_title = lesson.get("lesson_title", "unknown")
            lines.append(f"  Lesson {number}: {lesson_title}")

        # Expose the course as a source for the UI
        if course_link:
            self.last_sources = [
                f'<a href="{course_link}" target="_blank" rel="noopener noreferrer">{title}</a>'
            ]
        else:
            self.last_sources = [title]

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
