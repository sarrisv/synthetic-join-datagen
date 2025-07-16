import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type, Any, Union, Optional, Set

import numpy as np

from plan_structures import Plan

logger = logging.getLogger(__name__)


# --- ABCs and Concrete Implementations ---


# 1. Distribution Strategies (for data generation)
class DataDistribution(ABC):
    """Abstract base class for data value distributions"""

    @abstractmethod
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Generates a NumPy array of data values for a single Dask partition.
        """
        pass


class UniformDistribution(DataDistribution):
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        return rng.integers(low=1, high=domain_size + 1, size=values, dtype=np.int64)


class ZipfDistribution(DataDistribution):
    DEFAULT_SKEW: float = 2.0

    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist_args = dist_args or {}
        skew = dist_args.get("skew", self.DEFAULT_SKEW)
        if skew <= 1.0:
            logger.warning(
                f"Zipf distribution 'skew' must be > 1.0, but was {skew}. "
                f"Using default value of {self.DEFAULT_SKEW} instead."
            )
            skew = self.DEFAULT_SKEW
        arr_int64 = rng.zipf(a=skew, size=values).astype(np.int64)
        arr_clipped = np.clip(arr_int64, 1, domain_size)
        return arr_clipped


class NormalDistribution(DataDistribution):
    DEFAULT_MEAN_DIVISOR: float = 2.0
    DEFAULT_STD_DEV_DIVISOR: float = 5.0

    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist_args = dist_args or {}
        mean: float = dist_args.get("mean", domain_size / self.DEFAULT_MEAN_DIVISOR)
        std_dev: float = dist_args.get(
            "std_dev", domain_size / self.DEFAULT_STD_DEV_DIVISOR
        )
        clip_mode = dist_args.get("clip_mode", "center")

        arr_float = rng.normal(loc=mean, scale=std_dev, size=values)
        arr_int64 = np.round(arr_float).astype(np.int64)

        if clip_mode == "range":
            # Standard clipping to the specified domain [1, domain_size].
            arr_clipped = np.clip(arr_int64, 1, domain_size)
        else:  # Default to "center" mode.
            # This clipping creates a domain of a certain 'width' (domain_size)
            # centered around the 'mean', which may not start at 1.
            lower_bound = mean - (domain_size / 2)
            upper_bound = mean + (domain_size / 2)
            arr_clipped = np.clip(arr_int64, lower_bound, upper_bound)

        return arr_clipped


class JoinPattern(ABC):
    """Abstract base class for underlying pattern in the join"""

    def _create_join_tuple(
        self, rel1: str, rel2: str, attr_idx: int
    ) -> Tuple[str, str, int]:
        """Helper to create a canonically ordered join tuple"""
        canonical_order = sorted((rel1, rel2))
        return canonical_order[0], canonical_order[1], attr_idx

    @abstractmethod
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:  # Returns (Rel1_sorted, Rel2_sorted, AttrIndex)
        pass


class RandomPattern(JoinPattern):
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()

        if len(relations) < 2:
            return []

        unconnected = list(relations)
        connected = [unconnected.pop(rng.integers(len(unconnected)))]

        while unconnected:
            connected_rel = rng.choice(connected)
            unconnected_rel = unconnected.pop(rng.integers(len(unconnected)))
            joining_attr = int(rng.integers(num_attrs))
            joins_set.add(
                self._create_join_tuple(connected_rel, unconnected_rel, joining_attr)
            )
            connected.append(unconnected_rel)
        return sorted(list(joins_set))


class StarPattern(JoinPattern):
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()

        if len(relations) < 2:
            return []

        center_rel = relations[0]

        for other_rel in relations[1:]:
            joining_attr = int(rng.integers(num_attrs))
            joins_set.add(self._create_join_tuple(center_rel, other_rel, joining_attr))
        return sorted(list(joins_set))


class ChainPattern(JoinPattern):
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()

        if len(relations) < 2:
            return []

        for i in range(len(relations) - 1):
            curr_rel, next_rel = relations[i], relations[i + 1]
            joining_attr = int(rng.integers(num_attrs))
            joins_set.add(self._create_join_tuple(curr_rel, next_rel, joining_attr))
        return sorted(list(joins_set))


class CyclicPattern(ChainPattern):
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        if len(relations) < 2:
            return []

        # Get the linear chain of joins from the parent class
        chain_joins = super().generate_joins(rng, relations, num_attrs)
        joins_set = set(chain_joins)
        # Add the final join to close the cycle
        last_rel, first_rel = (
            relations[-1],
            relations[0],
        )
        joining_attr = int(rng.integers(num_attrs))
        joins_set.add(self._create_join_tuple(last_rel, first_rel, joining_attr))
        return sorted(list(joins_set))


# 3. Plan Formatters (for derived plan output)
class Formatter(ABC):
    """Abstract base class for formatting derived plan"""

    @abstractmethod
    def format(self, derived_plan: Plan) -> Union[str, Dict[str, Any]]:
        """Formats the derived plan"""
        pass


class TextFormatter(Formatter):
    def format(self, derived_plan: Plan) -> str:
        return derived_plan.to_text()


class JSONFormatter(Formatter):
    def format(self, derived_plan: Plan) -> Dict[str, Any]:
        """Formats the plan into a dictionary for JSON serialization"""
        return derived_plan.to_dict()


# --- Registries ---

DISTRIBUTIONS: Dict[str, Type[DataDistribution]] = {
    "uniform": UniformDistribution,
    "zipf": ZipfDistribution,
    "normal": NormalDistribution,
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
