# registries.py
"""
Defines Abstract Base Classes (ABCs) for extensible components like
data distributions, join patterns, and plan formatters, along with
their concrete implementations and registries for dynamic dispatch.
It also includes common dataclasses for plan structures.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type, Any, Union, Optional, Set, cast
from collections import defaultdict
import numpy as np
import random
from dataclasses import dataclass, field


# --- Common Dataclasses for Plan Structures ---
@dataclass
class DerivedPlanStage:
    """Base class for a stage in a derived plan."""

    pass


@dataclass
class TablePlanStage(DerivedPlanStage):
    """Represents a stage in a table-granularity plan,
    aggregating joins between two tables."""

    table1: str
    table2: str
    join_attributes: List[str]  # List of attribute names (e.g., ["attr1", "attr2"])


@dataclass
class AttributePlanStage(DerivedPlanStage):
    """Represents a stage in an attribute-granularity plan,
    focusing on a single attribute and all relations joining on it."""

    join_on_attribute: str  # Attribute name (e.g., "attr1")
    participating_relations: List[str]


@dataclass
class DerivedPlanOutput:
    """Holds the complete derived plan data ready for formatting."""

    plan_id: int
    join_pattern_name: str
    granularity: str
    selected_relations_in_scope: List[str]
    stages: List[Union[TablePlanStage, AttributePlanStage]]


# --- ABCs and Concrete Implementations ---


# 1. Distribution Strategies (for data generation)
class DistributionStrategy(ABC):
    """Abstract base class for data value distribution strategies."""

    @abstractmethod
    def generate_numpy_column(
        self,
        rng_numpy: np.random.Generator,
        num_values: int,
        domain_size: int,
        skew: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generates a NumPy array representing a column of data values.
        This is called per Dask partition.
        """
        pass


class UniformDistribution(DistributionStrategy):
    def generate_numpy_column(
        self,
        rng_numpy: np.random.Generator,
        num_values: int,
        domain_size: int,
        skew: Optional[float] = None,
    ) -> np.ndarray:
        return rng_numpy.integers(
            low=1, high=domain_size + 1, size=num_values, dtype=np.int64
        )


class ZipfDistribution(DistributionStrategy):
    def generate_numpy_column(
        self,
        rng_numpy: np.random.Generator,
        num_values: int,
        domain_size: int,
        skew: Optional[float] = 2.0,
    ) -> np.ndarray:
        actual_skew: float = skew if skew is not None and skew > 1.0 else 1.1
        arr_int64 = rng_numpy.zipf(a=actual_skew, size=num_values).astype(np.int64)
        arr_clipped = np.clip(arr_int64, 1, domain_size)
        return arr_clipped


class NormalDistribution(DistributionStrategy):
    def generate_numpy_column(
        self,
        rng_numpy: np.random.Generator,
        num_values: int,
        domain_size: int,
        skew: Optional[float] = None,
    ) -> np.ndarray:
        mean: float = domain_size / 2.0
        std_dev: float = domain_size / 5.0
        arr_float = rng_numpy.normal(loc=mean, scale=std_dev, size=num_values)
        arr_int64 = np.round(arr_float).astype(np.int64)
        arr_clipped = np.clip(arr_int64, 1, domain_size)
        return arr_clipped


# 2. Join Patterns (for fundamental plan generation)
class JoinPattern(ABC):
    """Abstract base class for patterns that form fundamental join connections."""

    @abstractmethod
    def generate_fundamental_joins(
        self,
        selected_relations_in_scope: List[str],
        num_attributes_per_relation: int,
        py_random_instance: random.Random,
    ) -> List[Tuple[str, str, int]]:  # Returns (Rel1_sorted, Rel2_sorted, AttrIndex)
        pass


class RandomJoinPattern(JoinPattern):
    def generate_fundamental_joins(
        self,
        selected_relations_in_scope: List[str],
        num_attributes: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        if len(selected_relations_in_scope) >= 2:
            temp_rels = list(selected_relations_in_scope)
            component = [temp_rels.pop(py_random.randrange(len(temp_rels)))]
            to_add = temp_rels[:]
            while to_add:
                r1 = py_random.choice(component)
                r2 = to_add.pop(py_random.randrange(len(to_add)))
                attr_idx = py_random.randint(0, num_attributes - 1)
                joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
                component.append(r2)
        return sorted(list(joins_set))


class StarJoinPattern(JoinPattern):
    def generate_fundamental_joins(
        self,
        selected_relations_in_scope: List[str],
        num_attributes: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        if len(selected_relations_in_scope) >= 2:
            center = selected_relations_in_scope[0]
            for other_rel in selected_relations_in_scope[1:]:
                attr_idx = py_random.randint(0, num_attributes - 1)
                joins_set.add(tuple(sorted((center, other_rel))) + (attr_idx,))
        return sorted(list(joins_set))


class ChainJoinPattern(JoinPattern):
    def generate_fundamental_joins(
        self,
        selected_relations_in_scope: List[str],
        num_attributes: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        for i in range(len(selected_relations_in_scope) - 1):
            r1, r2 = selected_relations_in_scope[i], selected_relations_in_scope[i + 1]
            attr_idx = py_random.randint(0, num_attributes - 1)
            joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
        return sorted(list(joins_set))


class CyclicJoinPattern(JoinPattern):
    def generate_fundamental_joins(
        self,
        selected_relations_in_scope: List[str],
        num_attributes: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        for i in range(len(selected_relations_in_scope) - 1):
            r1, r2 = selected_relations_in_scope[i], selected_relations_in_scope[i + 1]
            attr_idx = py_random.randint(0, num_attributes - 1)
            joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
        if len(selected_relations_in_scope) > 1:
            r_last, r_first = (
                selected_relations_in_scope[-1],
                selected_relations_in_scope[0],
            )
            attr_idx = py_random.randint(0, num_attributes - 1)
            joins_set.add(tuple(sorted((r_last, r_first))) + (attr_idx,))
        return sorted(list(joins_set))


# 3. Plan Formatters (for derived plan output)
class PlanFormatter(ABC):
    """Abstract base class for formatting derived plan data."""

    @abstractmethod
    def format_plan_body(
        self, derived_plan: DerivedPlanOutput
    ) -> Union[str, Dict[str, Any]]:
        """
        Formats the body/stages of the derived plan.
        Returns a string for text-based formats (body lines), or a dict for JSON (stages part).
        Common headers are added by the calling function for text formats.
        """
        pass


class TextPlanFormatter(PlanFormatter):
    def format_plan_body(self, derived_plan: DerivedPlanOutput) -> str:
        body_lines: List[str] = []
        if derived_plan.granularity == "table":
            body_lines.append(
                "# Table-at-a-Time Stages (Binary Joins - Aggregated by Pair):"
            )
            if not derived_plan.stages:
                body_lines.append("# (No binary join stages to derive)")
            for stage_obj in derived_plan.stages:
                stage = cast(TablePlanStage, stage_obj)
                body_lines.append(
                    f"{stage.table1},{stage.table2},{','.join(stage.join_attributes)}"
                )
        elif derived_plan.granularity == "attribute":
            body_lines.append(
                "# Attribute-at-a-Time Stages (Multi-way Joins per Attribute):"
            )
            if not derived_plan.stages:
                body_lines.append("# (No attribute stages to derive)")
            for stage_obj in derived_plan.stages:
                stage = cast(AttributePlanStage, stage_obj)
                body_lines.append(
                    f"{stage.join_on_attribute},{','.join(stage.participating_relations)}"
                )
        return "\n".join(body_lines)


class JSONPlanFormatter(PlanFormatter):
    def format_plan_body(self, derived_plan: DerivedPlanOutput) -> Dict[str, Any]:
        json_stages_list = []
        for stage in derived_plan.stages:
            if isinstance(stage, TablePlanStage):
                json_stages_list.append(
                    {
                        "table1": stage.table1,
                        "table2": stage.table2,
                        "join_attributes": stage.join_attributes,
                    }
                )
            elif isinstance(stage, AttributePlanStage):
                json_stages_list.append(
                    {
                        "join_on_attribute": stage.join_on_attribute,
                        "participating_relations": stage.participating_relations,
                    }
                )
        return {"stages": json_stages_list}


# --- Registries ---

DISTRIBUTION_STRATEGIES: Dict[str, Type[DistributionStrategy]] = {
    "uniform": UniformDistribution,
    "zipf": ZipfDistribution,
    "normal": NormalDistribution,
}

JOIN_PATTERNS: Dict[str, Type[JoinPattern]] = {
    "random": RandomJoinPattern,
    "star": StarJoinPattern,
    "chain": ChainJoinPattern,
    "cyclic": CyclicJoinPattern,
}

PLAN_FORMATTERS: Dict[str, Type[PlanFormatter]] = {
    "txt": TextPlanFormatter,
    "json": JSONPlanFormatter,
}
