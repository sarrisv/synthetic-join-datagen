# data_generation_module.py
"""
Handles the generation of data for a single relation using Dask for parallelism,
primarily leveraging dd.map_partitions for creating Dask DataFrames.
"""

import pathlib
import logging
import dask.dataframe as dd
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from registries import DISTRIBUTION_STRATEGIES, DistributionStrategy

logger = logging.getLogger(__name__)
INDENT: str = "  "


@dataclass
class RelationGenSpec:
    """Specification for generating a single data relation."""

    name: str
    num_attributes: int
    num_unique_tuples: int
    duplication_factor: int
    distribution_name: str
    skew: Optional[float]
    domain_size: int
    null_percentage: float
    output_format_name: str
    output_path_base: pathlib.Path
    seed: int  # This is the base seed for the relation
    num_data_partitions: int


def _generate_partition_data(
    df_metadata_placeholder: pd.DataFrame,
    total_unique_tuples: int,
    num_attributes: int,
    distribution_name: str,
    skew: Optional[float],
    domain_size: int,
    null_percentage: float,
    base_seed_for_relation: int,  # Renamed to match the calling context
    num_total_partitions: int,
    partition_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame for a single partition of unique data.
    This function is intended to be called by Dask's map_partitions.
    It derives a unique seed for this partition's execution.
    """
    part_logger = logging.getLogger(f"{__name__}._generate_partition_data")

    current_partition_index = 0
    if partition_info and "number" in partition_info:
        current_partition_index = partition_info["number"]
    else:
        part_logger.warning(
            "partition_info not available or 'number' key missing, assuming partition 0 for seed derivation."
        )

    # Calculate rows for this partition and PK offset
    base_rows_per_partition = total_unique_tuples // num_total_partitions
    remainder_rows = total_unique_tuples % num_total_partitions

    partition_num_rows = base_rows_per_partition + (
        1 if current_partition_index < remainder_rows else 0
    )

    pk_offset = (base_rows_per_partition * current_partition_index) + min(
        current_partition_index, remainder_rows
    )

    # part_logger.debug(f"Partition {current_partition_index}/{num_total_partitions}: "
    #                   f"base_seed {base_seed_for_relation}, offset_idx {current_partition_index}, "
    #                   f"{partition_num_rows} rows, PK offset {pk_offset}")

    if partition_num_rows == 0:
        # Return empty DataFrame matching meta if no rows for this partition
        return pd.DataFrame(columns=df_metadata_placeholder.columns).astype(
            df_metadata_placeholder.dtypes
        )

    # Derive a unique seed for this specific partition's data generation
    effective_partition_seed = base_seed_for_relation + current_partition_index + 1
    rng_numpy = np.random.default_rng(effective_partition_seed)
    distribution_impl = DISTRIBUTION_STRATEGIES[distribution_name]()

    data: Dict[str, np.ndarray] = {}
    # Primary Key (attr0)
    data["attr0"] = np.arange(pk_offset, pk_offset + partition_num_rows, dtype=np.int64)

    # Other attributes
    for i in range(1, num_attributes):
        attr_name = f"attr{i}"
        values_np = distribution_impl.generate_numpy_column(
            rng_numpy, partition_num_rows, domain_size, skew
        )

        # Apply nulls: Convert to float to allow np.nan
        values_float_np = values_np.astype(np.float64)
        null_mask = rng_numpy.random(size=partition_num_rows) < null_percentage
        values_float_np[null_mask] = np.nan
        data[attr_name] = values_float_np

    return pd.DataFrame(data, columns=list(df_metadata_placeholder.columns))


def generate_one_relation_dask_task(spec: RelationGenSpec) -> Optional[str]:
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
        if spec.num_attributes <= 0:
            task_logger.error(
                f"Cannot generate {spec.name} with {spec.num_attributes} attributes."
            )
            return None

        meta_columns = {
            f"attr{i}": (np.float64 if i > 0 else np.int64)
            for i in range(spec.num_attributes)
        }
        meta_df = pd.DataFrame(columns=list(meta_columns.keys())).astype(meta_columns)

        num_actual_partitions = max(
            1, spec.num_data_partitions if spec.num_unique_tuples > 0 else 1
        )

        # Create an empty Dask DataFrame with the desired number of partitions and meta
        # This defines the structure over which map_partitions will operate.
        empty_df_for_map = dd.from_pandas(meta_df, npartitions=num_actual_partitions)

        unique_ddf = empty_df_for_map.map_partitions(
            _generate_partition_data,  # Function to apply to each partition
            # Keyword arguments to pass to _generate_partition_data:
            total_unique_tuples=spec.num_unique_tuples,
            num_attributes=spec.num_attributes,
            distribution_name=spec.distribution_name,
            skew=spec.skew,
            domain_size=spec.domain_size,
            null_percentage=spec.null_percentage,
            base_seed_for_relation=spec.seed,  # Corrected: pass the base seed for the relation
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

        output_path_final: pathlib.Path = spec.output_path_base
        task_logger.info(
            f"Writing {spec.name} to {output_path_final} in {spec.output_format_name} format..."
        )

        if spec.output_format_name == "csv":
            ddf_to_write.to_csv(
                str(output_path_final), single_file=True, index=False, na_rep=""
            )
        elif spec.output_format_name == "json":
            ddf_to_write.to_json(
                str(output_path_final), orient="records", lines=False, single_file=True
            )
        elif spec.output_format_name == "parquet":
            final_parquet_path = output_path_final
            if not final_parquet_path.name.endswith(".parquet"):
                final_parquet_path = final_parquet_path.with_name(
                    final_parquet_path.name + ".parquet"
                )

            final_parquet_path.mkdir(parents=True, exist_ok=True)
            ddf_to_write.to_parquet(
                str(final_parquet_path),
                write_index=False,
                schema="infer",
                overwrite=True,
            )
            task_logger.info(
                f"Parquet data for {spec.name} written to directory: {final_parquet_path}"
            )
        else:
            task_logger.error(
                f"Unsupported data output format: {spec.output_format_name} for {spec.name}"
            )
            return None

        task_logger.info(f"Successfully generated and wrote relation: {spec.name}")
        return str(output_path_final)
    except Exception as e:
        task_logger.exception(f"Error in Dask task for relation {spec.name}: {e}")
        return None
