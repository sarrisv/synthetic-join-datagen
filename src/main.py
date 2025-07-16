import logging
from contextlib import contextmanager
from multiprocessing import cpu_count
from typing import List, Generator, Tuple, Dict, Any

import dask
from dask.distributed import Client, LocalCluster, progress, Client

from analysis_module import aggregate_individual_analyses
from data_generation import RelationSpec, generate_relation
from plan_generation import (
    PlanSpec,
    generate_plan,
    PlanConfig,
)
from cli import load_config, AppConfig

logger = logging.getLogger("datagen_main_orchestrator")


@contextmanager
def dask_cluster_context(config: AppConfig) -> Generator[Client, None, None]:
    """Context manager for Dask cluster lifecycle management"""
    cluster = None
    client = None

    try:
        dask_cluster_kwargs = {}
        if config.dask_workers is not None:
            dask_cluster_kwargs["n_workers"] = config.dask_workers
        if config.dask_threads_per_worker is not None:
            dask_cluster_kwargs["threads_per_worker"] = (
                config.dask_threads_per_worker
            )
        if config.dask_memory_limit != "auto":
            dask_cluster_kwargs["memory_limit"] = config.dask_memory_limit

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


def setup_output_directories(config: AppConfig) -> Tuple:
    data_output_path = config.base_output_dir / config.data_subdir
    plan_output_path = config.base_output_dir / config.plan_subdir
    analysis_output_path = config.base_output_dir / config.analysis_subdir
    try:
        data_output_path.mkdir(parents=True, exist_ok=True)
        if config.plans > 0 or config.generate_dot_visualization:
            plan_output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directories: {e}")
        raise
    return data_output_path, plan_output_path, analysis_output_path


def create_relation_specs(
    config: AppConfig, client: Client, data_output_path
) -> Tuple[List[str], List]:
    """Create relation specifications and return relation names and Dask tasks"""
    relation_names: List[str] = []
    data_tasks = []

    if config.relations <= 0:
        logger.info("No relations to generate (--relations is 0).")
        return relation_names, data_tasks

    logger.info("Preparing Dask tasks for data generation...")
    relation_names = [f"R{i}" for i in range(config.relations)]

    for i, r_name in enumerate(relation_names):
        rel_seed = config.seed + i + 101

        dist_args: Dict[str, Any] = {}
        if config.distribution == "zipf":
            if config.dist_skew is not None:
                dist_args["skew"] = config.dist_skew
        elif config.distribution == "normal":
            if config.dist_mean is not None:
                dist_args["mean"] = config.dist_mean
            if config.dist_std_dev is not None:
                dist_args["std_dev"] = config.dist_std_dev
            dist_args["clip_mode"] = config.dist_clip_mode

        output_base_name = r_name
        if config.data_output_format == "parquet":
            output_file_or_dir_path = data_output_path / f"{output_base_name}.parquet"
        else:  # csv, json
            output_file_or_dir_path = (
                data_output_path / f"{output_base_name}.{config.data_output_format}"
            )

        partitions_per_relation = config.dask_partitions_per_relation
        if (
            partitions_per_relation <= 0
        ):  # Default to number of workers if not specified or invalid
            partitions_per_relation = len(client.scheduler_info()["workers"]) if client else cpu_count()

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
        data_tasks.append(dask.delayed(generate_relation)(spec))

    return relation_names, data_tasks


def execute_data_generation(data_tasks: List) -> None:
    """Execute data generation tasks and handle results"""
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
    config: AppConfig, relation_names: List[str], plan_output_path, data_output_path
) -> List:
    """Create plan specifications and return Dask tasks"""
    plan_tasks = []

    logger.info("Preparing Dask tasks for plan generation...")

    relations_used = config.relations
    if config.max_join_relations is not None:
        if config.max_join_relations < 0:
            relations_used = 0
        else:
            relations_used = min(config.max_join_relations, config.relations)

    plan_gen_config = PlanConfig(
        plan_granularity=config.plan_granularity,
        plan_output_format=config.plan_output_format,
        generate_dot_visualization=config.generate_dot_visualization,
        num_attrs=config.attributes,
        run_on_the_fly_analysis=config.run_on_the_fly_analysis,
        data_output_path=data_output_path if config.run_on_the_fly_analysis else None,
        data_output_format=config.data_output_format,
        base_output_path=config.base_output_dir if config.run_on_the_fly_analysis else None,
        analysis_subdir=config.analysis_subdir,
    )

    plan_counter = 0

    for pattern_name in config.join_pattern:
        for _ in range(config.plans):
            plan_seed = config.seed + plan_counter * 1000 + 301

            spec = PlanSpec(
                plan_id=plan_counter,
                relations=list(relation_names),
                pattern=pattern_name,
                max_relations=relations_used,
                output_dir=plan_output_path,
                seed=plan_seed,
            )
            plan_tasks.append(dask.delayed(generate_plan)(spec, plan_gen_config))
            plan_counter += 1

    return plan_tasks


def execute_plan_generation(plan_tasks: List) -> None:
    """Execute plan generation tasks"""
    if not plan_tasks:
        return

    logger.info(f"Computing {len(plan_tasks)} Dask plan generation tasks...")
    plan_results = dask.compute(*plan_tasks)
    progress(plan_results)
    logger.info("Execution plan generation phase complete")


def aggregate_analysis_results(analysis_output_path) -> None:
    logger.info("Aggregating individual analysis files...")
    try:
        aggregate_individual_analyses(analysis_output_path)
        logger.info("Analysis aggregation complete")
    except Exception as e:
        logger.warning(f"Analysis aggregation failed: {e}")


def main() -> None:
    config = load_config()

    logger.info(f"Using global seed: {config.seed}")
    logger.info("Starting datagen.py with configuration:")
    for field_name, value in config.__dict__.items():
        logger.info(f"  {field_name}: {value}")
    logger.info("-" * 30)

    data_output_path, plan_output_path, analysis_output_path = setup_output_directories(
        config
    )

    with dask_cluster_context(config) as client:
        logger.info("Starting Data Generation...")
        relation_names, data_tasks = create_relation_specs(
            config, client, data_output_path
        )
        execute_data_generation(data_tasks)
        logger.info("-" * 30)

        if config.plans > 0:
            logger.info("Starting Plan Generation...")
            dask_plan_tasks = create_plan_specs(
                config, relation_names, plan_output_path, data_output_path
            )
            execute_plan_generation(dask_plan_tasks)
            logger.info("-" * 30)

    if config.plans > 0 and config.run_on_the_fly_analysis:
        logger.info("Starting Analysis Aggregation...")
        logger.info("Individual analyses were computed during plan generation.")
        aggregate_analysis_results(analysis_output_path)
        logger.info("-" * 30)

    logger.info("main.py finished")


if __name__ == "__main__":
    main()
