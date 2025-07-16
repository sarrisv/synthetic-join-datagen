import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Type, Any, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


# --- Data Distributions ---
# Abstract base class for data value distributions
class DataDistribution(ABC):
    @abstractmethod
    # Generates an array of integer values based on a distribution
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        pass


# Generates uniformly distributed integer values
class UniformDistribution(DataDistribution):
    # Generates an array of integer values based on a uniform distribution
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        # Generate integers from 1 up to and including domain_size
        return rng.integers(low=1, high=domain_size + 1, size=values, dtype=np.int64)


# Generates Zipf-distributed integer values (skewed)
class ZipfDistribution(DataDistribution):
    DEFAULT_SKEW: float = 2.0

    # Generates an array of integer values based on a Zipf distribution
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist_args = dist_args or {}  # Ensure dist_args is a dict
        # Get skew parameter or use default
        skew = dist_args.get("skew", self.DEFAULT_SKEW)
        # Zipf skew parameter 'a' must be > 1.0
        if skew <= 1.0:
            logger.warning(
                f"Zipf distribution 'skew' must be > 1.0, but was {skew}. "
                f"Using default value of {self.DEFAULT_SKEW} instead."
            )
            skew = self.DEFAULT_SKEW
        # Generate Zipf values and cast to int64
        arr_int64 = rng.zipf(a=skew, size=values).astype(np.int64)
        # Clip values to ensure they are within the domain [1, domain_size]
        arr_clipped = np.clip(arr_int64, 1, domain_size)
        return arr_clipped


# Generates normally distributed integer values
class NormalDistribution(DataDistribution):
    DEFAULT_STD_DEV_DIVISOR: float = 5.0

    # Generates an array of integer values based on a normal distribution
    def generate_values(
        self,
        rng: np.random.Generator,
        values: int,
        domain_size: int,
        dist_args: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        dist_args = dist_args or {}  # Ensure dist_args is a dict
        # Center the distribution in the middle of the domain
        mean: float = domain_size / 2.0
        # Get standard deviation or calculate a default
        std_dev: float = dist_args.get(
            "std_dev", domain_size / self.DEFAULT_STD_DEV_DIVISOR
        )
        arr_float = rng.normal(loc=mean, scale=std_dev, size=values)  # Generate normal float values
        arr_int64 = np.round(arr_float).astype(np.int64)  # Round to nearest integer
        # Clip values to ensure they are within the domain [1, domain_size]
        arr_clipped = np.clip(arr_int64, 1, domain_size)
        return arr_clipped


# --- Base Join Patterns ---
# Abstract base class for join graph patterns
class JoinPattern(ABC):
    @staticmethod
    def _create_join_tuple(rel1: str, rel2: str, attr_idx: int) -> Tuple[str, str, int]:
        # Create a canonically ordered join tuple for consistency
        canonical_order = sorted((rel1, rel2))
        return canonical_order[0], canonical_order[1], attr_idx

    @abstractmethod
    # Generates a list of joins based on a specific pattern
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:  # Returns canonical tuple
        pass


# Generates a random, connected join graph (a spanning tree)
class RandomPattern(JoinPattern):
    # Generates a random, connected join graph using a randomized Prim's-like algorithm
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()  # Use a set to store joins to avoid duplicates

        # Not enough relations or attributes to form a join
        if len(relations) < 2 or num_attrs <= 1:
            return []

        unconnected = list(relations)  # Start with all relations as unconnected
        # Move a random relation to the connected set to start the graph
        connected = [unconnected.pop(rng.integers(len(unconnected)))]

        # Iterate until all relations are connected
        while unconnected:
            # Pick a random relation that is already connected
            connected_rel = rng.choice(connected)
            # Pick and remove a random relation that is not yet connected
            unconnected_rel = unconnected.pop(rng.integers(len(unconnected)))

            # Select a random joining attribute (non-PK attributes)
            joining_attr = int(rng.integers(low=1, high=num_attrs))
            # Create the join tuple and add it to the set
            joins_set.add(
                self._create_join_tuple(connected_rel, unconnected_rel, joining_attr)
            )
            # The new relation is now part of the connected graph
            connected.append(unconnected_rel)
        return sorted(list(joins_set))  # Return a sorted list for deterministic output


# Generates a star join pattern with a central relation
class StarPattern(JoinPattern):
    # Generates a star join pattern, connecting all relations to a central one
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()  # Use a set to avoid duplicate joins

        # Not enough relations or attributes to form a join
        if len(relations) < 2 or num_attrs <= 1:
            return []

        # The first relation in the list is chosen as the center of the star
        center_rel = relations[0]

        # Join all other relations (dimensions) to the central relation (fact)
        for other_rel in relations[1:]:
            # Select a random joining attribute (non-PK attributes)
            joining_attr = int(rng.integers(low=1, high=num_attrs))
            joins_set.add(self._create_join_tuple(center_rel, other_rel, joining_attr))
        return sorted(list(joins_set))  # Return a sorted list for deterministic output


# Generates a chain/linear join pattern
class ChainPattern(JoinPattern):
    # Generates a chain join pattern, connecting relations sequentially
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        joins_set: Set[Tuple[str, str, int]] = set()  # Use a set to avoid duplicate joins

        # Not enough relations or attributes to form a join
        if len(relations) < 2 or num_attrs <= 1:
            return []

        # Iterate through adjacent pairs of relations to form the chain
        for i in range(len(relations) - 1):
            curr_rel, next_rel = relations[i], relations[i + 1]
            # Select a random joining attribute (non-PK attributes)
            joining_attr = int(rng.integers(low=1, high=num_attrs))
            joins_set.add(self._create_join_tuple(curr_rel, next_rel, joining_attr))
        return sorted(list(joins_set))  # Return a sorted list for deterministic output


# Generates a cyclic join pattern by connecting the ends of a chain
class CyclicPattern(ChainPattern):
    # Generates a cyclic join pattern by creating a chain and joining the ends
    def generate_joins(
        self,
        rng: np.random.Generator,
        relations: List[str],
        num_attrs: int,
    ) -> List[Tuple[str, str, int]]:
        # First, create a chain pattern as the base
        chain_joins = super().generate_joins(rng, relations, num_attrs)
        # If no chain joins were created, we can't make a cycle
        if not chain_joins:
            return []

        joins_set = set(chain_joins)
        # Get the first and last relations in the chain to close the loop
        last_rel, first_rel = (relations[-1], relations[0])

        # Select a random joining attribute (non-PK attributes)
        joining_attr = int(rng.integers(low=1, high=num_attrs))
        # Add the final join to create the cycle
        joins_set.add(self._create_join_tuple(last_rel, first_rel, joining_attr))
        return sorted(list(joins_set))  # Return a sorted list for deterministic output


# --- Registries ---
# Maps distribution names to their respective classes
DISTRIBUTIONS: Dict[str, Type[DataDistribution]] = {
    "uniform": UniformDistribution,
    "zipf": ZipfDistribution,
    "normal": NormalDistribution,
}
# Maps join pattern names to their respective classes
JOIN_PATTERNS: Dict[str, Type[JoinPattern]] = {
    "random": RandomPattern,
    "star": StarPattern,
    "chain": ChainPattern,
    "cyclic": CyclicPattern,
}
