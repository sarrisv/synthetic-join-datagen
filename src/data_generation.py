import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Any

import dask.dataframe as dd
import numpy as np
import pandas as pd

from registries import DISTRIBUTIONS

logger = logging.getLogger(__name__)
# Logger for the partition generation function, defined once at module level.
_partition_logger = logging.getLogger(f"{__name__}._generate_partition")
INDENT: str = "  "


@dataclass
class RelationSpec:
    """Specification for generating a single data relation."""

    name: str
    num_attrs: int
    unique_tuples: int
    duplication_factor: int
    distribution: str
    dist_args: Optional[Dict[str, Any]]
    domain_size: int
    null_percentage: float
    format: str
    output_path_base: pathlib.Path
    seed: int  # This is the base seed for the relation
    partitions: int


def _generate_partition(
    meta_df: pd.DataFrame,
    unique_tuples: int,
    num_attrs: int,
    distribution: str,
    dist_args: Optional[Dict[str, Any]],
    domain_size: int,
    null_percentage: float,
    base_seed: int,
    num_total_partitions: int,
    partition_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame for a single partition of unique data.
    This function is intended to be called by Dask's map_partitions.
    It derives a unique seed for this partition's execution.
    """

    if partition_info and "number" in partition_info:
        partition_idx = partition_info["number"]
    else:
        _partition_logger.warning(
            "partition_info not available or 'number' key missing, assuming partition 0 for seed derivation."
        )
        partition_idx = 0

    # Calculate the number of rows this partition is responsible for generating
    rows_per_part, remainder = divmod(unique_tuples, num_total_partitions)
    partition_num_rows = rows_per_part + (1 if partition_idx < remainder else 0)
    pk_offset = (rows_per_part * partition_idx) + min(partition_idx, remainder)

    if partition_num_rows == 0:
        # Return empty DataFrame matching meta if no rows for this partition
        return pd.DataFrame(columns=meta_df.columns).astype(meta_df.dtypes)

    # Derive a unique seed for this specific partition's data generation
    seed = base_seed + partition_idx + 1
    rng_numpy = np.random.default_rng(seed)
    distribution_impl = DISTRIBUTIONS[distribution]()

    # Primary Key (attr0)
    data: Dict[str, np.ndarray] = {"attr0": np.arange(pk_offset, pk_offset + partition_num_rows, dtype=np.int64)}

    # Other attributes
    for i in range(1, num_attrs):
        attr_name = f"attr{i}"
        # Generate as float to hold np.nan, which is the standard way to represent
        # a missing value before casting to a nullable integer type like pd.Int64Dtype.
        values_np = distribution_impl.generate_values(
            rng_numpy, partition_num_rows, domain_size, dist_args
        ).astype(np.float64)

        # Apply nulls using np.nan. Dask will convert this to pd.NA when it
        # enforces the nullable integer dtype from the 'meta' object.
        if null_percentage > 0:
            null_mask = rng_numpy.random(size=partition_num_rows) < null_percentage
            values_np[null_mask] = np.nan
        data[attr_name] = values_np

    # Create DataFrame. Columns with nulls (np.nan) will have float dtype.
    df = pd.DataFrame(data, columns=list(meta_df.columns))

    # Enforce the target dtypes from metadata. This correctly converts float
    # columns with np.nan to nullable integer columns (pd.NA).
    return df.astype(meta_df.dtypes)


def generate_relation(spec: RelationSpec) -> Optional[str]:
    """
    Dask task to generate and write a single relation file using dd.map_partitions.

    Returns:
        The final output path (file or directory) on success, None on failure.
    """
    task_logger = logging.getLogger(f"{__name__}.{spec.name}")
    task_logger.info(
        f"Starting generation for: {spec.name}, Output: {spec.output_path_base}"
    )

    try:
        if spec.num_attrs <= 0:
            task_logger.error(
                f"Cannot generate {spec.name} with {spec.num_attrs} attributes."
            )
            return None

        # Define the DataFrame schema. The primary key 'attr0' is a standard, non-nullable integer.
        # Other attributes use Pandas' nullable integer type to correctly handle missing values.
        meta_columns = {
            f"attr{i}": (np.int64 if i == 0 else pd.Int64Dtype())
            for i in range(spec.num_attrs)
        }
        meta_df = pd.DataFrame(columns=list(meta_columns.keys())).astype(meta_columns)

        num_actual_partitions = max(1, spec.partitions if spec.unique_tuples > 0 else 1)

        # Create an empty Dask DataFrame with the desired number of partitions and meta
        # This defines the structure over which map_partitions will operate.
        empty_df_for_map = dd.from_pandas(meta_df, npartitions=num_actual_partitions)

        unique_ddf = empty_df_for_map.map_partitions(
            _generate_partition,  # Function to apply to each partition
            # Keyword arguments to pass to _generate_partition_data:
            unique_tuples=spec.unique_tuples,
            num_attrs=spec.num_attrs,
            distribution=spec.distribution,
            dist_args=spec.dist_args,
            domain_size=spec.domain_size,
            null_percentage=spec.null_percentage,
            base_seed=spec.seed,
            num_total_partitions=num_actual_partitions,
            meta=meta_df,
        )

        if spec.duplication_factor > 1:
            # Apply duplication per partition
            ddf_to_write = unique_ddf.map_partitions(
                lambda part_df: pd.concat(
                    [part_df] * spec.duplication_factor, ignore_index=True
                )
                if not part_df.empty
                else part_df,
                meta=meta_df,
            )
        else:
            ddf_to_write = unique_ddf

        output_path: pathlib.Path = spec.output_path_base
        task_logger.info(
            f"Writing {spec.name} to {output_path} in {spec.format} format..."
        )

        if spec.format == "csv":
            ddf_to_write.to_csv(
                str(output_path), single_file=True, index=False, na_rep=""
            )
        elif spec.format == "json":
            ddf_to_write.to_json(
                str(output_path), orient="records", lines=False, single_file=True
            )
        elif spec.format == "parquet":
            # Dask's to_parquet handles directory creation. The output_path is
            # already correctly formed to be a directory name (e.g., '.../data/R0.parquet').
            ddf_to_write.to_parquet(
                str(output_path),
                write_index=False,
                schema="infer",
                overwrite=True,
            )
            task_logger.info(
                f"Parquet data for {spec.name} written to directory: {output_path}"
            )
        else:
            task_logger.error(
                f"Unsupported data output format: {spec.format} for {spec.name}"
            )
            return None

        task_logger.info(f"Successfully generated and wrote relation: {spec.name}")
        return str(output_path)
    except Exception as e:
        task_logger.exception(f"Error in Dask task for relation {spec.name}: {e}")
        return None
