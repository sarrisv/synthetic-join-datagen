import logging
import pathlib
from contextlib import contextmanager
from multiprocessing import cpu_count
from typing import List, Generator, Tuple, Dict, Any

import dask
from dask.distributed import LocalCluster, Client
from dask.diagnostics import ProgressBar

from analysis_module import aggregate_individual_analyses
from cli import ArgsConfig, load_config
from data_generation import RelationSpec, generate_relation
from plan_generation import (
    PlanSpec,
    generate_plan,
    PlanConfig,
)

logger = logging.getLogger("main")


# Set up and tear down a Dask cluster for computation
@contextmanager
def _dask_cluster_context(config: ArgsConfig) -> Generator[Client, None, None]:
    """Context manager for Dask cluster lifecycle management"""
    cluster = None
    client = None

    # Useful for efficient shuffle operations like in value_counts()
    # This setting enables peer-to-peer shuffling, which is often faster
    dask_config = {
        "distributed.p2p.enabled": True,
        "dataframe.shuffle.method": "p2p",
    }

    try:
        dask_cluster_kwargs = {}
        # Build Dask cluster configuration from command-line arguments
        if config.dask_workers is not None:
            dask_cluster_kwargs["n_workers"] = config.dask_workers
        if config.dask_threads_per_worker is not None:
            dask_cluster_kwargs["threads_per_worker"] = (
                config.dask_threads_per_worker
            )
        if config.dask_memory_limit != "auto":
            dask_cluster_kwargs["memory_limit"] = config.dask_memory_limit

        logger.info(
            f"Initializing Dask LocalCluster with derived config: {dask_cluster_kwargs}"
        )

        cluster = LocalCluster(config=dask_config, **dask_cluster_kwargs)
        client = Client(cluster)

        logger.info(f"Dask client started. Dashboard: {client.dashboard_link}")
        logger.info("-" * 30)
        yield client # Yield the client to the 'with' block

    except Exception as e:
        logger.error(
            f"Failed to start Dask client/cluster: {e}. This program requires Dask to function."
        )
        raise
    finally:
        # Ensure cluster and client are always shut down cleanly
        if client:
            try:
                logger.info("Shutting down Dask client and cluster...")
                client.close()
                if cluster:
                    cluster.close(timeout=10)
                logger.info("Dask client and cluster shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down Dask client/cluster: {e}")


# Prepare the output directory structure for an iteration
def _setup_output_directories(base_dir: pathlib.Path, config: ArgsConfig) -> Tuple:
    """Creates the necessary subdirectories within a given base directory"""
    data_output_path = base_dir / config.data_subdir
    plan_output_path = base_dir / config.plan_subdir
    analysis_output_path = base_dir / config.analysis_subdir
    try:
        # Conditionally create plan and analysis directories if needed
        data_output_path.mkdir(parents=True, exist_ok=True)
        if config.plans > 0 or config.gen_dot_viz:
            plan_output_path.mkdir(parents=True, exist_ok=True)
        if config.plans > 0 and config.analyze:
            analysis_output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directories: {e}")
        raise
    return data_output_path, plan_output_path, analysis_output_path


# Generates specifications for each relation to be created
def _create_relation_specs(
    config: ArgsConfig, client: Client, data_output_path
) -> Tuple[List[str], List]:
    """Create relation specifications and return relation names and Dask tasks"""
    relation_names: List[str] = []
    data_tasks = []

    if config.relations <= 0:
        logger.warning("No relations to generate (--relations is 0).")
        return relation_names, data_tasks

    logger.info("Preparing Dask tasks for data generation...")
    relation_names = [f"R{i}" for i in range(config.relations)]

    # Create a spec for each relation
    for i, r_name in enumerate(relation_names):
        rel_seed = config.seed + i + 101 # Use a unique seed for each relation

        # Configure distribution-specific arguments
        dist_args: Dict[str, Any] = {}
        if config.distribution == "zipf":
            if config.dist_skew is not None:
                dist_args["skew"] = config.dist_skew
        elif config.distribution == "normal":
            if config.dist_std_dev is not None:
                dist_args["std_dev"] = config.dist_std_dev

        # Define the output path for the relation data
        output_file_or_dir_path = (
            data_output_path / f"{r_name}.{config.data_output_format}"
        )

        partitions_per_relation = config.dask_partitions_per_relation
        if partitions_per_relation <= 0: # Default to # of workers if not specified or invalid
            partitions_per_relation = (
                len(client.scheduler_info()["workers"]) if client else cpu_count()
            )

        # Assemble the specification for a single relation
        spec = RelationSpec(
            name=r_name,
            num_attrs=config.attributes,
            unique_tuples=config.unique_tuples,
            duplication_factor=config.duplication_factor,
            distribution=config.distribution,
            dist_args=dist_args,
            domain_size=config.domain_size,
            null_percentage=config.null_percentage,
            format=config.data_output_format,
            output_path_base=output_file_or_dir_path,
            seed=rel_seed,
            partitions=partitions_per_relation,
        )
        # Create a Dask delayed task for generating this relation
        data_tasks.append(generate_relation(spec))

    return relation_names, data_tasks


# Generates specifications for each query plan to be created
def _create_plan_specs(
    config: ArgsConfig,
    relation_names: List[str],
    plan_output_path: pathlib.Path,
    plan_gen_config: PlanConfig,
    data_dependencies: List,
) -> List:
    """Create plan specifications and return Dask tasks"""
    plan_tasks = []

    logger.info("Preparing Dask tasks for plan generation...")

    # Determine how many relations to use in each join plan
    if config.max_join_relations is None:
        relations_used = config.relations
    else:
        # Clamp to be non-negative, then take the minimum with total relations
        relations_used = min(max(0, config.max_join_relations), config.relations)

    plan_counter = 0

    # Generate plans for each specified join pattern
    for pattern_name in config.join_pattern:
        for _ in range(config.plans):
            plan_seed = config.seed + plan_counter * 1000 + 301 # Unique seed per plan

            # Assemble the specification for a single query plan
            spec = PlanSpec(
                plan_id=plan_counter,
                relations=list(relation_names),
                pattern=pattern_name,
                max_relations=relations_used,
                output_dir=plan_output_path,
                seed=plan_seed,
            )
            # Create a Dask delayed task for generating this plan
            # The plan task may depend on data generation tasks if analysis is enabled
            plan_tasks.append(
                dask.delayed(generate_plan)(
                    spec, plan_gen_config, analysis_deps=data_dependencies
                )
            )
            plan_counter += 1

    return plan_tasks


# Combines individual JSON analysis files into a single aggregated file
def _aggregate_analysis_results(analysis_output_path) -> None:
    try:
        aggregate_individual_analyses(
            analysis_output_path
        )  # This is from an external module, so no underscore
        logger.info("Analysis aggregation complete")
    except Exception as e:
        logger.warning(f"Analysis aggregation failed: {e}")


# Orchestrates a single data generation and processing iteration
def _run_iteration(
    config: ArgsConfig,
    client: Client,
    iteration_name: str,
    iteration_base_dir: pathlib.Path,
    iteration_num: int,
    total_iterations: int,
) -> None:
    """Runs a single data generation iteration"""
    logger.info("=" * 80)
    logger.info(
        f"STARTING ITERATION: '{iteration_name}' ({iteration_num}/{total_iterations})"
    )
    logger.info("=" * 80)

    logger.info(f"Using seed for this iteration: {config.seed}")
    logger.info(f"Output for this iteration will be in: {iteration_base_dir}")
    logger.info("Configuration for this iteration:")
    for field_name, value in config.__dict__.items():
        logger.info(f"  {field_name}: {value}")
    logger.info("-" * 30)

    # Create directories for the current iteration
    data_output_path, plan_output_path, analysis_output_path = (
        _setup_output_directories(iteration_base_dir, config)
    )

    # --- Data Generation ---
    # First, define the tasks for creating the base relations
    relation_names, data_tasks = _create_relation_specs(
        config, client, data_output_path
    )

    # --- Plan Generation ---
    plan_tasks = []
    if config.plans > 0: # Only create plans if requested
        # Create the plan configuration here, in the main orchestration function.
        plan_gen_config = PlanConfig(
            plan_granularity=config.plan_granularity,
            plan_output_format=config.plan_output_format,
            gen_dot_viz=config.gen_dot_viz,
            num_attrs=config.attributes,
            analyze=config.analyze,
            data_output_path=data_output_path if config.analyze else None,
            base_output_path=iteration_base_dir if config.analyze else None,
            analysis_subdir=config.analysis_subdir,
        )

        plan_tasks = _create_plan_specs(
            config,
            relation_names,
            plan_output_path,
            plan_gen_config,
            data_dependencies=data_tasks if config.analyze else [], # Pass data tasks as dependencies if analysis is enabled
        )

    # --- Combined Execution ---
    # Compute final tasks; plan_tasks will trigger data_tasks via dependency
    final_tasks = plan_tasks or data_tasks
    if final_tasks:
        logger.info(
            f"Computing a combined Dask graph of {len(final_tasks)} final-stage tasks..."
        )
        with ProgressBar():
            dask.compute(*final_tasks)
        logger.info("Data and Plan generation phase complete.")
        logger.info("-" * 30)

    # --- Analysis Aggregation ---
    # After all individual analyses are done, combine them into one file
    if config.plans > 0 and config.analyze:
        logger.info("Starting Analysis Aggregation...")
        _aggregate_analysis_results(analysis_output_path)
        logger.info("-" * 30)


def main() -> None:
    # Load one or more configurations from the config file
    configs = load_config()
    if not configs:
        logger.warning("No configurations loaded or an error occurred. Exiting.")
        return

    is_multi_run = len(configs) > 1
    # Store the original base path for multi-runs to create named sub-folders
    original_base_output_dir = configs[0].base_output_dir if configs else None
    total_iterations = len(configs)

    logger.info(f"Starting datagen.py. Found {total_iterations} generation task(s).")
    logger.info("-" * 30)

    # A single Dask cluster is configured from the first config and shared across all iterations
    with _dask_cluster_context(configs[0]) as client:
        # Loop through each configuration and run an iteration
        for i, config in enumerate(configs):
            iteration_name = config.name or f"iteration_{i}" # Use config name or a default

            # Determine the correct base directory for this iteration
            iteration_base_dir = config.base_output_dir
            if is_multi_run and original_base_output_dir is not None:
                # In multi-run mode, create a subdirectory for each named configuration
                iteration_base_dir = original_base_output_dir / iteration_name

            _run_iteration(
                config,
                client,
                iteration_name,
                iteration_base_dir,
                i + 1,
                total_iterations,
            )

    logger.info("main.py finished all iterations.")


if __name__ == "__main__":
    main()
