import logging
from contextlib import contextmanager
from multiprocessing import cpu_count
from typing import List, Generator, Tuple

import dask
from dask.distributed import Client, LocalCluster, progress

from analysis_module import aggregate_individual_analyses, analyze_all_plans
from data_generation import RelationSpec, generate_relation
from plan_generation import (
    PlanSpec,
    generate_plan,
    PlanConfig,
)
from setup import load_config

logger = logging.getLogger("datagen_main_orchestrator")


@contextmanager
def dask_cluster_context(global_args) -> Generator[Client, None, None]:
    """Context manager for Dask cluster lifecycle management."""
    cluster = None
    client = None

    try:
        dask_cluster_kwargs = {}
        if global_args.dask_workers is not None:
            dask_cluster_kwargs["n_workers"] = global_args.dask_workers
        if global_args.dask_threads_per_worker is not None:
            dask_cluster_kwargs["threads_per_worker"] = (
                global_args.dask_threads_per_worker
            )
        if global_args.dask_memory_limit != "auto":
            dask_cluster_kwargs["memory_limit"] = global_args.dask_memory_limit

        logger.info(
            f"Initializing Dask LocalCluster with kwargs: {dask_cluster_kwargs}"
        )
        cluster = LocalCluster(**dask_cluster_kwargs)
        client = Client(cluster)
        logger.info(f"Dask client started. Dashboard: {client.dashboard_link}")
        logger.info("-" * 30)

        yield client

    except Exception as e:
        logger.error(
            f"Failed to start Dask client/cluster: {e}. This program requires Dask to function."
        )
        raise
    finally:
        if client:
            try:
                logger.info("Shutting down Dask client and cluster...")
                client.close()
                if cluster:
                    cluster.close(timeout=10)
                logger.info("Dask client and cluster shut down.")
            except Exception as e:
                logger.warning(f"Error shutting down Dask client/cluster: {e}")


def setup_output_directories(global_args) -> Tuple:
    data_output_path = global_args.base_output_dir / global_args.data_subdir_name
    plan_output_path = global_args.base_output_dir / global_args.plan_subdir_name
    analysis_output_path = (
        global_args.base_output_dir / global_args.analysis_subdir_name
    )
    try:
        data_output_path.mkdir(parents=True, exist_ok=True)
        if global_args.plans > 0 or global_args.generate_dot_visualization:
            plan_output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directories: {e}")
        raise
    return data_output_path, plan_output_path, analysis_output_path


def create_relation_specs(
    global_args, client: Client, data_output_path
) -> Tuple[List[str], List]:
    """Create relation specifications and return relation names and Dask tasks."""
    relation_names: List[str] = []
    data_tasks = []

    if global_args.relations <= 0:
        logger.info("No relations to generate (--relations is 0).")
        return relation_names, data_tasks

    logger.info("Preparing Dask tasks for data generation...")
    relation_names = [f"R{i}" for i in range(global_args.relations)]

    for i, r_name in enumerate(relation_names):
        rel_seed = global_args.seed + i + 101

        output_base_name = r_name
        if global_args.data_output_format == "parquet":
            output_file_or_dir_path = data_output_path / f"{output_base_name}.parquet"
        else:  # csv, json
            output_file_or_dir_path = (
                data_output_path
                / f"{output_base_name}.{global_args.data_output_format}"
            )

        # Determine number of partitions
        partitions_per_relation = global_args.dask_partitions_per_relation
        if (
            partitions_per_relation <= 0
        ):  # Default to number of workers if not specified or invalid
            partitions_per_relation = (
                len(client.scheduler_info()["workers"]) if client else cpu_count()
            )

        spec = RelationSpec(
            name=r_name,
            num_attrs=global_args.attributes,
            unique_tuples=global_args.unique_tuples,
            dup_factor=global_args.dup_factor,
            distribution=global_args.distribution,
            skew=global_args.skew,
            domain_size=global_args.domain_size,
            null_pct=global_args.null_pct,
            format=global_args.data_output_format,
            output_path_base=output_file_or_dir_path,
            seed=rel_seed,
            partitions=partitions_per_relation,
        )
        data_tasks.append(dask.delayed(generate_relation)(spec))

    return relation_names, data_tasks


def execute_data_generation(data_tasks: List) -> None:
    """Execute data generation tasks and handle results."""
    if not data_tasks:
        return

    logger.info(f"Computing {len(data_tasks)} data generation tasks...")
    computed_results = dask.compute(*data_tasks)
    progress(computed_results)

    for res_path in computed_results:
        if res_path:
            logger.debug(f"Data task completed, output: {res_path}")
        else:
            logger.error(
                "A data generation task may have failed (returned None). Check worker logs."
            )

    logger.info("Data generation phase complete.")


def create_plan_specs(
    global_args, relation_names: List[str], plan_output_path, data_output_path
) -> List:
    """Create plan specifications and return Dask tasks."""
    plan_tasks = []

    logger.info("Preparing Dask tasks for plan generation...")

    # Determine actual max relations in scope
    relations_used = global_args.relations
    if global_args.max_join_relations is not None:
        if global_args.max_join_relations < 0:
            relations_used = 0
        else:
            relations_used = min(global_args.max_join_relations, global_args.relations)

    # Create global plan arguments subset
    config = PlanConfig(
        plan_granularity=global_args.plan_granularity,
        plan_output_format=global_args.plan_output_format,
        generate_dot_visualization=global_args.generate_dot_visualization,
        num_attrs=global_args.attributes,
        run_on_the_fly_analysis=global_args.run_on_the_fly_analysis,
        data_output_path=data_output_path
        if global_args.run_on_the_fly_analysis
        else None,
        data_output_format=global_args.data_output_format,
        base_output_path=global_args.base_output_dir
        if global_args.run_on_the_fly_analysis
        else None,
    )

    # Generate plan tasks
    plan_counter = 0

    for pattern_name in global_args.join_pattern:
        for _ in range(global_args.plans):
            plan_seed = global_args.seed + plan_counter * 1000 + 301

            spec = PlanSpec(
                plan_id=plan_counter,
                all_relation_names=list(relation_names),
                pattern=pattern_name,
                max_relations=relations_used,
                output_dir=plan_output_path,
                seed_for_plan=plan_seed,
            )
            plan_tasks.append(dask.delayed(generate_plan)(spec, config))
            plan_counter += 1

    return plan_tasks


def execute_plan_generation(plan_tasks: List) -> None:
    """Execute plan generation tasks."""
    if not plan_tasks:
        return

    logger.info(f"Computing {len(plan_tasks)} Dask plan generation tasks...")
    plan_results = dask.compute(*plan_tasks)
    progress(plan_results)
    logger.info("Execution plan generation phase complete")


def execute_analysis_aggregation(plan_output_path, analysis_output_path) -> None:
    logger.info("Aggregating individual analysis files...")
    try:
        aggregate_individual_analyses(analysis_output_path, plan_output_path)
        logger.info("Analysis aggregation complete")
    except Exception as e:
        logger.warning(f"Analysis aggregation failed: {e}")


def execute_batch_analysis(
    global_args, plan_output_path, data_output_path, analysis_output_path
) -> None:
    logger.info("Running batch selectivity analysis on generated data and plans...")
    try:
        analyses = analyze_all_plans(
            plan_output_path,
            data_output_path,
            global_args.data_output_format,
            analysis_output_path,
        )
        logger.info(f"Batch analysis complete, processed {len(analyses)} plans")
        execute_analysis_aggregation(plan_output_path, analysis_output_path)

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")


def main() -> None:
    global_args = load_config()

    logger.info(f"Using global seed: {global_args.seed}")
    logger.info("Starting datagen.py with configuration:")
    for field_name, value in global_args.__dict__.items():
        logger.info(f"  {field_name}: {value}")
    logger.info("-" * 30)

    data_output_path, plan_output_path, analysis_output_path = setup_output_directories(
        global_args
    )

    with dask_cluster_context(global_args) as client:
        logger.info("Starting Data Generation...")
        relation_names, data_tasks = create_relation_specs(
            global_args, client, data_output_path
        )
        execute_data_generation(data_tasks)
        logger.info("-" * 30)

        if global_args.plans > 0:
            logger.info("Starting Plan Generation...")
            dask_plan_tasks = create_plan_specs(
                global_args, relation_names, plan_output_path, data_output_path
            )
            execute_plan_generation(dask_plan_tasks)
            logger.info("-" * 30)

    if global_args.plans > 0:
        if global_args.run_on_the_fly_analysis:
            logger.info("Starting Analysis...")
            logger.info("Individual analyses computed during plan generation")
            execute_analysis_aggregation(plan_output_path, analysis_output_path)
        elif global_args.run_analysis:
            logger.info("Starting Analysis...")
            execute_batch_analysis(
                global_args, plan_output_path, data_output_path, analysis_output_path
            )
        logger.info("-" * 30)

    logger.info("main.py finished")


if __name__ == "__main__":
    main()
