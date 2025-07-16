from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Union, Dict, Any


@dataclass
class Stage(ABC):
    # Abstract base class for a single stage in a query plan
    @abstractmethod
    def to_text(self) -> str:
        # Abstract method to convert the stage to a text representation
        pass


@dataclass
class BinaryJoinStage(Stage):
    # Represents a join stage between two relations
    rel1: str
    rel2: str
    join_attributes: List[str]

    def to_text(self) -> str:
        # Converts the binary join stage to a comma-separated string
        return f"{self.rel1},{self.rel2},{','.join(self.join_attributes)}"


@dataclass
class MultiwayJoinStage(Stage):
    # Represents a multi-way join stage on a single attribute
    joining_attribute: str
    relations: List[str]

    def to_text(self) -> str:
        # Converts the multi-way join stage to a comma-separated string
        return f"{self.joining_attribute},{','.join(self.relations)}"


@dataclass
class Plan:
    # Represents a full query execution plan
    plan_id: int
    pattern: str
    granularity: str
    relations: List[str]
    stages: Union[List[BinaryJoinStage], List[MultiwayJoinStage]]

    def to_dict(self) -> Dict[str, Any]:
        # Converts the Plan object to a dictionary
        return asdict(self)

    def to_text(self) -> str:
        # Converts the Plan object to a human-readable text format
        header_lines = [
            f"# Execution Plan {self.plan_id}",
            f"# Pattern for Base Joins: {self.pattern}",
            f"# Output Granularity: {self.granularity}",
            f"# Relations in Plan Scope: {', '.join(self.relations) if self.relations else 'None'}",
        ]

        # Build the body of the plan text based on granularity
        body_lines = []
        if self.granularity == "table":
            body_lines.append("# Table-at-a-Time Stages:")
        elif self.granularity == "attribute":
            body_lines.append("# Attribute-at-a-Time Stages:")

        # Add stages to the body, or a note if there are none
        if not self.stages:
            body_lines.append(f"# (No {self.granularity} stages to derive)")
        else:
            body_lines.extend(stage.to_text() for stage in self.stages)

        return "\n".join(header_lines) + "\n" + "\n".join(body_lines)
