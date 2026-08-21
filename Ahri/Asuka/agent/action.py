from typing import Any

from pydantic import BaseModel, Field


class Action(BaseModel):
    name: str = Field(description="tool name")
    args: dict[str, Any] | None = Field(description="tool input arguments")
