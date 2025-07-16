import json
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from analysis_module import AttributeMetadata, RelationMetadata

from registries import DISTRIBUTIONS

logger = logging.getLogger("datagen")
_partition_logger = logging.getLogger(f"datagen._generate_partition")


@dataclass
class RelationSpec:
    """Specification for generating a single data relation"""

    name: str  # The name of the relation
    num_attrs: int  # Total number of attributes (columns) in the relation
    unique_tuples: int  # Number of unique rows to generate before duplication
    duplication_factor: int  # Factor by which to duplicate the unique rows
    distribution: str  # Name of the data distribution for attribute values
    dist_args: Optional[Dict[str, Any]]  # Arguments for the chosen distribution
    domain_size: int  # The upper bound for generated values (exclusive)
    null_percentage: float  # The percentage of nulls to introduce in attributes
    format: str  # The output file format (e.g., 'csv', 'parquet')
    output_path_base: pathlib.Path  # The base path for the output file or directory
    seed: int  # This is the base seed for the relation
    partitions: int  # The number of Dask partitions for generation


def _generate_partition(
    meta_df: pd.DataFrame,
    unique_tuples: int,
    num_attrs: int,
    distribution: str,
    dist_args: Optional[Dict[str, Any]],
    domain_size: int,
    null_percentage: float,
    seed: int,
    num_total_partitions: int,
    partition_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame for a single partition of unique data.
    This function is intended to be called by Dask's map_partitions.
    It derives a unique seed for this partition's execution.
    """

    # Dask provides partition info, including the partition index ('number')
    if partition_info and "number" in partition_info:
        partition_idx = partition_info["number"]
    else:
        # Fallback for cases where partition info is not available
        _partition_logger.warning(
            "partition_info not available or 'number' key missing, assuming partition 0 for seed derivation."
        )
        partition_idx = 0

    # Calculate base number of rows per partition and any remainder
    rows_per_part, remainder = divmod(unique_tuples, num_total_partitions)

    # Distribute the remainder rows across the first 'remainder' partitions
    partition_num_rows = rows_per_part + (1 if partition_idx < remainder else 0)
    # Calculate the starting primary key value for this partition to ensure global uniqueness
    pk_offset = (rows_per_part * partition_idx) + min(partition_idx, remainder)

    if partition_num_rows == 0:
        # Return an empty DataFrame with the correct schema if no rows are needed
        return pd.DataFrame(columns=meta_df.columns).astype(meta_df.dtypes)

    # Derive a unique, deterministic seed for this partition's data generation
    seed = seed + partition_idx + 1
    rng_numpy = np.random.default_rng(seed)
    distribution_impl = DISTRIBUTIONS[distribution]()

    # Primary Key (attr0) is a simple, non-null, unique integer sequence
    data: Dict[str, np.ndarray] = {
        "attr0": np.arange(pk_offset, pk_offset + partition_num_rows, dtype=np.int64)
    }

    # Generate data for all other attributes
    for i in range(1, num_attrs):
        attr_name = f"attr{i}"
        # Generate as float to accommodate np.nan for nulls before final casting
        values_np = distribution_impl.generate_values(
            rng_numpy, partition_num_rows, domain_size, dist_args
        ).astype(np.float64)
        if null_percentage > 0:
            # Create a boolean mask to inject nulls (as np.nan)
            null_mask = rng_numpy.random(size=partition_num_rows) < null_percentage
            values_np[null_mask] = np.nan
        data[attr_name] = values_np

    df = pd.DataFrame(data, columns=list(meta_df.columns))

    # Enforce the target dtypes, converting float columns with np.nan to nullable integers (pd.NA)
    return df.astype(meta_df.dtypes)


@dask.delayed
def _write_relation_metadata(
    spec: RelationSpec,
    total_rows: int,
    # This will unpack the list of stats collections
    *stats_collections: Tuple[pd.Series, int],
) -> None:
    """
    A Dask-delayed task that receives computed statistics and writes the metadata file.
    Dask computes the collections passed as arguments before executing this task.
    """
    task_logger = logging.getLogger(f"{__name__}.{spec.name}")
    # Define the path for the output metadata JSON file
    metadata_path = spec.output_path_base.with_suffix(".meta.json")
    task_logger.info(f"Assembling and writing metadata to {metadata_path.name}")

    try:
        # This dictionary will hold the metadata for each attribute
        attribute_metadata: Dict[str, AttributeMetadata] = {}

        # The stats_collections is a flat tuple: (vc_series, null_count, vc_series, null_count, ...)
        # We need the column names to rebuild the per-attribute stats
        columns = [f"attr{i}" for i in range(spec.num_attrs)]

        for i, col in enumerate(columns):
            # Unpack the value counts and null count for the current column
            value_counts = stats_collections[i * 2]
            null_count = stats_collections[i * 2 + 1]
            # Convert value_counts (pandas Series) to a dict of str:int for JSON serialization
            value_counts_dict = {
                str(k): int(v) for k, v in value_counts.to_dict().items()
            }
            attribute_metadata[col] = AttributeMetadata(
                null_count=int(null_count),  # Total nulls in the column
                distinct_values=len(
                    value_counts_dict
                ),  # Total distinct non-null values
                value_counts=value_counts_dict,
            )

        # Assemble the final relation metadata object
        relation_meta = RelationMetadata(
            name=spec.name, total_rows=total_rows, attributes=attribute_metadata
        )
        # Write the metadata to a JSON file
        metadata_path.write_text(json.dumps(asdict(relation_meta), indent=2))
    except Exception as e:
        task_logger.error(f"Failed to assemble or write metadata for {spec.name}: {e}")


def generate_relation(spec: RelationSpec) -> Optional[str]:
    """
    Builds a Dask graph to generate a relation, write it to disk, and save its metadata.
    """
    task_logger = logging.getLogger(f"{__name__}.{spec.name}")
    task_logger.info(
        f"Starting generation for: {spec.name}, Output: {spec.output_path_base}"
    )

    try:
        # Basic validation on the number of attributes
        if spec.num_attrs <= 0:
            task_logger.error(
                f"Cannot generate {spec.name} with {spec.num_attrs} attributes."
            )
            return None

        # Define the DataFrame schema: 'attr0' is non-nullable, others are nullable integers
        meta_columns = {
            f"attr{i}": (np.int64 if i == 0 else pd.Int64Dtype())
            for i in range(spec.num_attrs)
        }
        # Create an empty DataFrame with the target schema to provide metadata to Dask
        meta_df = pd.DataFrame(columns=list(meta_columns.keys())).astype(meta_columns)

        # Ensure there's at least one partition, even for empty relations
        num_actual_partitions = max(1, spec.partitions if spec.unique_tuples > 0 else 1)

        # Create an empty Dask DataFrame to define the structure for map_partitions
        empty_df_for_map = dd.from_pandas(meta_df, npartitions=num_actual_partitions)

        # This is the main data generation step, creating unique rows in a Dask DataFrame
        unique_ddf = empty_df_for_map.map_partitions(
            _generate_partition,
            unique_tuples=spec.unique_tuples,  # Total unique rows to generate
            num_attrs=spec.num_attrs,  # Number of attributes
            distribution=spec.distribution,  # Data distribution profile
            dist_args=spec.dist_args,  # Arguments for the distribution
            domain_size=spec.domain_size,  # Upper bound for values
            null_percentage=spec.null_percentage,  # Percentage of nulls to inject
            seed=spec.seed,  # Base seed for reproducibility
            num_total_partitions=num_actual_partitions,  # Total partition count
            meta=meta_df,  # DataFrame metadata
        )

        if spec.duplication_factor > 1:
            # If duplication is requested, apply it per partition
            ddf_to_write = unique_ddf.map_partitions(
                lambda part_df: pd.concat(
                    [part_df] * spec.duplication_factor, ignore_index=True
                )
                if not part_df.empty  # Avoid errors on empty partitions
                else part_df,  # Return the empty partition as-is
                meta=meta_df,  # Preserve metadata
            )
        else:
            # If no duplication, the DataFrame to write is just the unique rows
            ddf_to_write = unique_ddf

        # --- Graph-based data writing and metadata generation ---
        output_path: pathlib.Path = spec.output_path_base
        task_logger.info(
            f"Writing {spec.name} to {output_path} in {spec.format} format..."
        )

        # Create a delayed task for writing the data to disk based on the format
        if spec.format == "csv":
            data_write_delayed = ddf_to_write.to_csv(
                str(output_path), single_file=True, index=False, na_rep=""
            )
        elif spec.format == "json":
            # Note: single_file=True for JSON can be memory-intensive
            data_write_delayed = ddf_to_write.to_json(
                str(output_path), orient="records", lines=False, single_file=True
            )
        elif spec.format == "parquet":
            # Dask's to_parquet handles directory creation; output_path is the directory name
            data_write_delayed = ddf_to_write.to_parquet(
                str(output_path), write_index=False, schema="infer", overwrite=True
            )
        else:
            task_logger.error(
                f"Unsupported data output format: {spec.format} for {spec.name}"
            )
            return None

        # Create delayed tasks to compute statistics for each column for the metadata
        stats_collections = []
        for col in ddf_to_write.columns:
            stats_collections.append(
                ddf_to_write[col].value_counts()
            )  # Distinct value counts
            stats_collections.append(
                ddf_to_write[col].isnull().sum()
            )  # Total null count
        # Create the delayed task for writing the computed metadata
        metadata_write_delayed = _write_relation_metadata(
            spec, len(ddf_to_write), *stats_collections
        )

        # This final delayed task depends on both data writing and metadata writing
        @dask.delayed
        def final_marker(data_write_result, _meta_write_signal):
            # The _meta_write_signal argument creates a dependency on metadata writing
            # Its value (None) is not used, it just enforces execution order
            task_logger.info(f"Successfully generated and wrote relation: {spec.name}")
            # Return the output path as a success indicator
            return str(spec.output_path_base)

        # Return the final Dask task that wires everything together
        return final_marker(data_write_delayed, metadata_write_delayed)

    except Exception as e:
        task_logger.exception(f"Error in Dask task for relation {spec.name}: {e}")
        return None
