import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Type, Any, Union, Optional, Set, cast

import numpy as np


# --- Common Dataclasses for Plan Structures ---
@dataclass
class Stage:
    """Base class for a stage in a derived plan."""

    pass


@dataclass
class TableStage(Stage):
    """Represents a stage in a table-granularity plan,
    aggregating joins between two tables."""

    table1: str
    table2: str
    join_attributes: List[str]  # List of attribute names (e.g., ["attr1", "attr2"])


@dataclass
class AttributeStage(Stage):
    """Represents a stage in an attribute-granularity plan,
    focusing on a single attribute and all relations joining on it."""

    join_on_attribute: str  # Attribute name (e.g., "attr1")
    participating_relations: List[str]


@dataclass
class Plan:
    """Holds the complete derived plan data ready for formatting."""

    plan_id: int
    pattern: str
    granularity: str
    relations: List[str]
    stages: List[Union[TableStage, AttributeStage]]


# --- ABCs and Concrete Implementations ---


# 1. Distribution Strategies (for data generation)
class DataDistribution(ABC):
    """Abstract base class for data value distribution strategies."""

    @abstractmethod
    def generate_values(
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


class Uniform(DataDistribution):
    def generate_values(
        self,
        rng_numpy: np.random.Generator,
        num_values: int,
        domain_size: int,
        skew: Optional[float] = None,
    ) -> np.ndarray:
        return rng_numpy.integers(
            low=1, high=domain_size + 1, size=num_values, dtype=np.int64
        )


class Zipf(DataDistribution):
    def generate_values(
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


class Normal(DataDistribution):
    def generate_values(
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


class JoinPattern(ABC):
    """Abstract base class for patterns that form fundamental join connections."""

    @abstractmethod
    def generate_joins(
        self,
        relations: List[str],
        num_attrs: int,
        py_random_instance: random.Random,
    ) -> List[Tuple[str, str, int]]:  # Returns (Rel1_sorted, Rel2_sorted, AttrIndex)
        pass


class RandomPattern(JoinPattern):
    def generate_joins(
        self,
        relations: List[str],
        num_attrs: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        if len(relations) >= 2:
            temp_rels = list(relations)
            component = [temp_rels.pop(py_random.randrange(len(temp_rels)))]
            to_add = temp_rels[:]
            while to_add:
                r1 = py_random.choice(component)
                r2 = to_add.pop(py_random.randrange(len(to_add)))
                attr_idx = py_random.randint(0, num_attrs - 1)
                joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
                component.append(r2)
        return sorted(list(joins_set))


class StarPattern(JoinPattern):
    def generate_joins(
        self,
        relations: List[str],
        num_attrs: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        if len(relations) >= 2:
            center = relations[0]
            for other_rel in relations[1:]:
                attr_idx = py_random.randint(0, num_attrs - 1)
                joins_set.add(tuple(sorted((center, other_rel))) + (attr_idx,))
        return sorted(list(joins_set))


class ChainPattern(JoinPattern):
    def generate_joins(
        self,
        relations: List[str],
        num_attrs: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        for i in range(len(relations) - 1):
            r1, r2 = relations[i], relations[i + 1]
            attr_idx = py_random.randint(0, num_attrs - 1)
            joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
        return sorted(list(joins_set))


class CyclicPattern(JoinPattern):
    def generate_joins(
        self,
        relations: List[str],
        num_attrs: int,
        py_random: random.Random,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()
        for i in range(len(relations) - 1):
            r1, r2 = relations[i], relations[i + 1]
            attr_idx = py_random.randint(0, num_attrs - 1)
            joins_set.add(tuple(sorted((r1, r2))) + (attr_idx,))
        if len(relations) > 1:
            r_last, r_first = (
                relations[-1],
                relations[0],
            )
            attr_idx = py_random.randint(0, num_attrs - 1)
            joins_set.add(tuple(sorted((r_last, r_first))) + (attr_idx,))
        return sorted(list(joins_set))


# 3. Plan Formatters (for derived plan output)
class Formatter(ABC):
    """Abstract base class for formatting derived plan data."""

    @abstractmethod
    def format(self, derived_plan: Plan) -> Union[str, Dict[str, Any]]:
        """
        Formats the body/stages of the derived plan.
        Returns a string for text-based formats (body lines), or a dict for JSON (stages part).
        Common headers are added by the calling function for text formats.
        """
        pass


class TextFormatter(Formatter):
    def format(self, derived_plan: Plan) -> str:
        body_lines: List[str] = []
        if derived_plan.granularity == "table":
            body_lines.append(
                "# Table-at-a-Time Stages (Binary Joins - Aggregated by Pair):"
            )
            if not derived_plan.stages:
                body_lines.append("# (No binary join stages to derive)")
            for stage_obj in derived_plan.stages:
                stage = cast(TableStage, stage_obj)
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
                stage = cast(AttributeStage, stage_obj)
                body_lines.append(
                    f"{stage.join_on_attribute},{','.join(stage.participating_relations)}"
                )
        return "\n".join(body_lines)


class JSONFormatter(Formatter):
    def format(self, derived_plan: Plan) -> Dict[str, Any]:
        json_stages_list = []
        for stage in derived_plan.stages:
            if isinstance(stage, TableStage):
                json_stages_list.append(
                    {
                        "table1": stage.table1,
                        "table2": stage.table2,
                        "join_attributes": stage.join_attributes,
                    }
                )
            elif isinstance(stage, AttributeStage):
                json_stages_list.append(
                    {
                        "join_on_attribute": stage.join_on_attribute,
                        "participating_relations": stage.participating_relations,
                    }
                )
        return {"stages": json_stages_list}


# --- Registries ---

DISTRIBUTIONS: Dict[str, Type[DataDistribution]] = {
    "uniform": Uniform,
    "zipf": Zipf,
    "normal": Normal,
}

JOIN_PATTERNS: Dict[str, Type[JoinPattern]] = {
    "random": RandomPattern,
    "star": StarPattern,
    "chain": ChainPattern,
    "cyclic": CyclicPattern,
}

FORMATTERS: Dict[str, Type[Formatter]] = {
    "txt": TextFormatter,
    "json": JSONFormatter,
}
