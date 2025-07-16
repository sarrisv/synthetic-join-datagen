from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union, Dict, Any


@dataclass
class Stage(ABC):
    """Base class for a stage in a derived plan."""

    @abstractmethod
    def to_text(self) -> str:
        """Formats the stage as a single line of text for the plan file."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Returns the dictionary representation for this stage."""
        pass


@dataclass
class TableStage(Stage):
    """A binary join stage in a table-at-a-time plan."""

    rel1: str
    rel2: str
    join_attributes: List[str]  # List of attribute names (e.g., ["attr1", "attr2"])

    def to_text(self) -> str:
        return f"{self.rel1},{self.rel2},{','.join(self.join_attributes)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel1": self.rel1,
            "rel2": self.rel2,
            "join_attributes": self.join_attributes,
        }


@dataclass
class AttributeStage(Stage):
    """A multi-way join stage on a single attribute."""

    joining_attribute: str  # Attribute name (e.g., "attr1")
    participating_relations: List[str]

    def to_text(self) -> str:
        return f"{self.joining_attribute},{','.join(self.participating_relations)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joining_attribute": self.joining_attribute,
            "participating_relations": self.participating_relations,
        }


@dataclass
class Plan:
    """Holds the complete derived plan data."""

    plan_id: int
    pattern: str
    granularity: str
    relations: List[str]
    stages: List[Union[TableStage, AttributeStage]]

    def to_dict(self) -> Dict[str, Any]:
        """Returns the complete dictionary representation of the plan."""
        return {
            "plan_id": self.plan_id,
            "pattern": self.pattern,
            "granularity": self.granularity,
            "relations_in_scope": self.relations,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_text(self) -> str:
        """Formats the entire plan into a text file representation, including a header."""
        header_lines = [
            f"# Execution Plan {self.plan_id}",
            f"# Pattern for Base Joins: {self.pattern}",
            f"# Output Granularity: {self.granularity}",
            f"# Relations in Plan Scope: {', '.join(self.relations) if self.relations else 'None'}",
        ]

        body_lines = []
        if self.granularity == "table":
            body_lines.append("# Table-at-a-Time Stages:")
        elif self.granularity == "attribute":
            body_lines.append("# Attribute-at-a-Time Stages:")

        if not self.stages:
            body_lines.append(f"# (No {self.granularity} stages to derive)")
        else:
            body_lines.extend(stage.to_text() for stage in self.stages)

        return "\n".join(header_lines) + "\n" + "\n".join(body_lines)
