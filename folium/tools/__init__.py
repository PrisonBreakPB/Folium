"""Tool registry."""

from .bash import BashTool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool
from .todo import TodoTool
from .web import WebFetchTool, WebSearchTool
from .pdf import PdfFetchTool
from .paper_search import PaperSearchTool
from .paper_validate import PaperValidateTool
from .arxiv_search import ArxivSearchTool
from .session_history import SessionHistoryTool
from .memory import MemoryTool

ALL_TOOLS = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    TodoTool(),
    WebSearchTool(),
    WebFetchTool(),
    PdfFetchTool(),
    PaperSearchTool(),
    PaperValidateTool(),
    ArxivSearchTool(),
    SessionHistoryTool(),
    MemoryTool(),
]


def create_tools():
    """Create a fresh default tool set for one Agent instance."""
    return [
        BashTool(),
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GlobTool(),
        GrepTool(),
        AgentTool(),
        TodoTool(),
        WebSearchTool(),
        WebFetchTool(),
        PdfFetchTool(),
        PaperSearchTool(),
        PaperValidateTool(),
        ArxivSearchTool(),
        SessionHistoryTool(),
        MemoryTool(),
    ]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
