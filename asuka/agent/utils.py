from __future__ import annotations

import sys
from typing import Any
from uuid import UUID

from colorama import Fore, Style
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.BLUE
RETURN_COLOR = Fore.CYAN
CODE_COLOR = Fore.WHITE


def color_print(text, color=None, end="\n"):
    context = color + text + Style.RESET_ALL + end if color is not None else text + end
    sys.stdout.write(context)
    sys.stdout.flush()


class ColoredPrintHandler(BaseCallbackHandler):

    def __init__(self, color: str) -> None:
        super().__init__()
        self.color = color

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        color_print(token, self.color, end="")
        return token

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        color_print("\n", self.color, end="")
        return response

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        print()
        color_print("\n[tool Return]", RETURN_COLOR)
        color_print(output, OBSERVATION_COLOR)
        return output

    def on_thought_start(self, index: int, **kwargs: Any):
        color_print(f"\n[Thought: {index}]", ROUND_COLOR)
        return index
