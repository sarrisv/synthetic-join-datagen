import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Any

import dask.dataframe as dd
import numpy as np
import pandas as pd

from registries import DISTRIBUTIONS

logger = logging.getLogger(__name__)
INDENT: str = "  "


@dataclass
class RelationSpec:
    """Specification for generating a single data relation."""

    name: str
    num_attrs: int
    unique_tuples: int
    dup_factor: int
    distribution: str
    skew: Optional[float]
    domain_size: int
    null_pct: float
    format: str
    output_path_base: pathlib.Path
    seed: int  # This is the base seed for the relation
    partitions: int


def _generate_partition(
    df_metadata_placeholder: pd.DataFrame,
    unique_tuples: int,
    num_attrs: int,
    distribution: str,
    skew: Optional[float],
    domain_size: int,
    null_pct: float,
    base_seed: int,  # Renamed to match the calling context
    num_total_partitions: int,
    partition_info: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame for a single partition of unique data.
    This function is intended to be called by Dask's map_partitions.
    It derives a unique seed for this partition's execution.
    """
    part_logger = logging.getLogger(f"{__name__}._generate_partition_data")

    partition_idx = 0
    if partition_info and "number" in partition_info:
        partition_idx = partition_info["number"]
    else:
        part_logger.warning(
            "partition_info not available or 'number' key missing, assuming partition 0 for seed derivation."
        )

    # Calculate rows for this partition and PK offset
    base_rows_per_partition = unique_tuples // num_total_partitions
    remainder_rows = unique_tuples % num_total_partitions

    partition_num_rows = base_rows_per_partition + (
        1 if partition_idx < remainder_rows else 0
    )

    pk_offset = (base_rows_per_partition * partition_idx) + min(
        partition_idx, remainder_rows
    )

    # part_logger.debug(f"Partition {partition_idx}/{num_total_partitions}: "
    #                   f"base_seed {base_seed}, offset_idx {partition_idx}, "
    #                   f"{partition_num_rows} rows, PK offset {pk_offset}")

    if partition_num_rows == 0:
        # Return empty DataFrame matching meta if no rows for this partition
        return pd.DataFrame(columns=df_metadata_placeholder.columns).astype(
            df_metadata_placeholder.dtypes
        )

    # Derive a unique seed for this specific partition's data generation
    seed = base_seed + partition_idx + 1
    rng_numpy = np.random.default_rng(seed)
    distribution_impl = DISTRIBUTIONS[distribution]()

    data: Dict[str, np.ndarray] = {
        "attr0": np.arange(pk_offset, pk_offset + partition_num_rows, dtype=np.int64)
    }
    # Primary Key (attr0)

    # Other attributes
    for i in range(1, num_attrs):
        attr_name = f"attr{i}"
        values_np = distribution_impl.generate_values(
            rng_numpy, partition_num_rows, domain_size, skew
        )

        # Apply nulls
        null_mask = rng_numpy.random(size=partition_num_rows) < null_pct
        values_np[null_mask] = -1
        data[attr_name] = values_np

        # Version w/ actual nulls but 2x storage
        # values_float_np = values_np.astype(np.float64)
        # values_float_np[null_mask] = np.nan
        # data[attr_name] = values_float_np

    return pd.DataFrame(data, columns=list(df_metadata_placeholder.columns))


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

        meta_columns = {f"attr{i}": np.int64 for i in range(spec.num_attrs)}

        # Version w/ actual nulls but 2x storage
        # meta_columns = {
        #     f"attr{i}": (np.float64 if i > 0 else np.int64)
        #     for i in range(spec.num_attrs)
        # }

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
            skew=spec.skew,
            domain_size=spec.domain_size,
            null_pct=spec.null_pct,
            base_seed=spec.seed,
            num_total_partitions=num_actual_partitions,
            meta=meta_df,
        )

        if spec.dup_factor > 1:
            # Apply duplication per partition
            ddf_to_write = unique_ddf.map_partitions(
                lambda part_df: pd.concat(
                    [part_df] * spec.dup_factor, ignore_index=True
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
                str(output_path), single_file=True, index=False, na_rep="-1"
            )
        elif spec.format == "json":
            ddf_to_write.to_json(
                str(output_path), orient="records", lines=False, single_file=True
            )
        elif spec.format == "parquet":
            final_parquet_path = output_path
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
                f"Unsupported data output format: {spec.format} for {spec.name}"
            )
            return None

        task_logger.info(f"Successfully generated and wrote relation: {spec.name}")
        return str(output_path)
    except Exception as e:
        task_logger.exception(f"Error in Dask task for relation {spec.name}: {e}")
        return None
