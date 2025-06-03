# main.py
"""
Main orchestrator for the Data Generation CLI.
Parses arguments, sets up Dask, and dispatches tasks for data and plan generation.
"""

import argparse
import pathlib
import random
import time
import logging
import dask
import tomllib
from dask.distributed import Client, LocalCluster, progress
from multiprocessing import cpu_count
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, cast

from data_generation import RelationGenSpec, generate_one_relation_dask_task
from plan_generation import (
    FundamentalPlanSpecForWorker,
    process_one_fundamental_plan_task,
    GlobalCLIArgsPlanSubset,
)
from registries import DISTRIBUTION_STRATEGIES, JOIN_PATTERNS, PLAN_FORMATTERS

INDENT: str = "  "
logger = logging.getLogger("datagen_main_orchestrator")


@dataclass
class GlobalCLIArgs:
    relations: int
    attributes: int
    unique_tuples: int
    duplication_factor: int
    distribution: str
    skew: Optional[float]
    domain_size: int
    null_percentage: float
    data_output_format: str
    plans: int
    join_pattern: List[str]
    plan_granularity: List[str]
    max_join_relations: Optional[int]
    plan_output_format: List[str]
    generate_dot_visualization: bool
    seed: int
    base_output_dir: pathlib.Path
    data_subdir_name: str
    plan_subdir_name: str
    verbose: bool
    log_file: Optional[pathlib.Path]
    dask_workers: Optional[int]
    dask_threads_per_worker: Optional[int]
    dask_memory_limit: Optional[str]
    dask_num_data_partitions_per_relation: int
    run_analysis: bool
    run_on_the_fly_analysis: bool


def load_config_file(config_path: pathlib.Path) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    try:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)
        logger.info(f"Loaded configuration from: {config_path}")
        return config if config else {}
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise


def merge_config_with_args(config: Dict[str, Any], parsed_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config file values with CLI arguments. CLI args take precedence."""
    merged = {}

    # Start with config file values - flatten TOML tables
    for section, values in config.items():
        if isinstance(values, dict):
            # Handle TOML tables (data_generation, plan_generation, etc.)
            for key, value in values.items():
                # For dask table, prefix with dask_
                if section == "dask":
                    arg_key = f"dask_{key.replace('-', '_')}"
                else:
                    arg_key = key.replace('-', '_')
                merged[arg_key] = value
        else:
            # Handle top-level keys
            arg_key = section.replace('-', '_')
            merged[arg_key] = values

    # Override with CLI arguments (exclude default values by checking if they differ from defaults)
    args_dict = vars(parsed_args)
    parser_defaults = {
        'relations': 3, 'attributes': 3, 'unique_tuples': 1000, 'duplication_factor': 1,
        'distribution': 'normal', 'skew': None, 'domain_size': 1000, 'null_percentage': 0.0,
        'data_output_format': 'csv', 'plans': 1, 'join_pattern': ['random'],
        'plan_granularity': ['table'], 'max_join_relations': None, 'plan_output_format': ['txt'],
        'generate_dot_visualization': False, 'seed': None, 'base_output_dir': 'generated_output',
        'data_subdir': 'data', 'plan_subdir': 'plans', 'verbose': False, 'log_file': None,
        'dask_workers': None, 'dask_threads_per_worker': None, 'dask_memory_limit': 'auto',
        'dask_num_data_partitions_per_relation': 0, 'run_analysis': False,
        'run_on_the_fly_analysis': False
    }

    for key, value in args_dict.items():
        if key == 'config':  # Skip the config file argument itself
            continue
        # Only override if the CLI value differs from the default
        if key in parser_defaults and value != parser_defaults[key]:
            merged[key] = value
        elif key not in parser_defaults and value is not None:
            merged[key] = value

    return merged


def create_global_args_from_merged(merged: Dict[str, Any]) -> GlobalCLIArgs:
    """Create GlobalCLIArgs from merged configuration."""
    # Handle special cases and defaults
    actual_skew = merged.get('skew')
    if merged.get('distribution') == 'zipf' and actual_skew is None:
        actual_skew = 2.0

    seed = merged.get('seed')
    if seed is None:
        seed = random.randint(1, 2**32 - 1)

    return GlobalCLIArgs(
        relations=merged.get('relations', 3),
        attributes=merged.get('attributes', 3),
        unique_tuples=merged.get('unique_tuples', 1000),
        duplication_factor=merged.get('duplication_factor', 1),
        distribution=merged.get('distribution', 'normal'),
        skew=actual_skew,
        domain_size=merged.get('domain_size', 1000),
        null_percentage=merged.get('null_percentage', 0.0),
        data_output_format=merged.get('data_output_format', 'csv'),
        plans=merged.get('plans', 1),
        join_pattern=merged.get('join_pattern', ['random']),
        plan_granularity=merged.get('plan_granularity', ['table']),
        max_join_relations=merged.get('max_join_relations'),
        plan_output_format=merged.get('plan_output_format', ['txt']),
        generate_dot_visualization=merged.get('generate_dot_visualization', False),
        seed=seed,
        base_output_dir=pathlib.Path(merged.get('base_output_dir', 'generated_output')),
        data_subdir_name=merged.get('data_subdir', 'data'),
        plan_subdir_name=merged.get('plan_subdir', 'plans'),
        verbose=merged.get('verbose', False),
        log_file=pathlib.Path(merged['log_file']) if merged.get('log_file') else None,
        dask_workers=merged.get('dask_workers'),
        dask_threads_per_worker=merged.get('dask_threads_per_worker'),
        dask_memory_limit=merged.get('dask_memory_limit', 'auto'),
        dask_num_data_partitions_per_relation=merged.get('dask_num_data_partitions_per_relation', 0),
        run_analysis=merged.get('run_analysis', False),
        run_on_the_fly_analysis=merged.get('run_on_the_fly_analysis', False),
    )


def setup_logging(is_verbose: bool, log_file_path: Optional[pathlib.Path]) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_level = logging.DEBUG if is_verbose else logging.INFO

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-35s - %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(console_formatter)

    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.DEBUG)

    if log_file_path:
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file_path, mode="w")
            fh.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)-35s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s"
            )
            fh.setFormatter(file_formatter)
            logging.getLogger().addHandler(fh)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file_path}: {e}")


def run_datagen() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset and Execution Plan Generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("Data Generation Parameters")
    data_group.add_argument(
        "--relations", type=int, default=3, metavar="N", help="Number of relations."
    )
    data_group.add_argument(
        "--attributes",
        type=int,
        default=3,
        metavar="M",
        help="Attributes per relation (attr0 is PK).",
    )
    data_group.add_argument(
        "--unique-tuples",
        type=int,
        default=1000,
        metavar="U",
        help="Unique tuples per relation.",
    )
    data_group.add_argument(
        "--duplication-factor",
        type=int,
        default=1,
        metavar="D",
        help="Duplication factor (>=1).",
    )
    data_group.add_argument(
        "--distribution",
        choices=list(DISTRIBUTION_STRATEGIES.keys()),
        default="normal",
        help="Distribution for non-PKs.",
    )
    data_group.add_argument(
        "--skew",
        type=float,
        default=None,
        metavar="S",
        help="Skew 'a' for Zipf (>1.0). Default: 2.0 if Zipf.",
    )
    data_group.add_argument(
        "--domain-size",
        type=int,
        default=1000,
        metavar="MAX_V",
        help="Max value for non-PKs [1, MAX_V].",
    )
    data_group.add_argument(
        "--null-percentage",
        type=float,
        default=0.0,
        metavar="P_N",
        help="Null percentage [0.0, 1.0) for non-PKs.",
    )
    data_group.add_argument(
        "--data-output-format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Format for data files.",
    )

    plan_group = parser.add_argument_group("Execution Plan Generation Parameters")
    plan_group.add_argument(
        "--plans",
        type=int,
        default=1,
        metavar="N_P_PAT",
        help="Fundamental plans per join pattern.",
    )
    plan_group.add_argument(
        "--join-pattern",
        nargs="+",
        choices=list(JOIN_PATTERNS.keys()),
        default=["random"],
        metavar="PAT",
        help="Fundamental join pattern/s.",
    )
    plan_group.add_argument(
        "--plan-granularity",
        nargs="+",
        choices=["table", "attribute"],
        default=["table"],
        metavar="GRAN",
        help="Output plan granularity/ies.",
    )
    plan_group.add_argument(
        "--max-join-relations",
        type=int,
        default=None,
        metavar="MAX_R",
        help="Max relations in plan scope (default: all).",
    )
    plan_group.add_argument(
        "--plan-output-format",
        nargs="+",
        choices=list(PLAN_FORMATTERS.keys()),
        default=["txt"],
        metavar="PLAN_FMT",
        help="Format(s) for derived plan files.",
    )
    plan_group.add_argument(
        "--generate-dot-visualization",
        action="store_true",
        help="Generate DOT viz of fundamental joins.",
    )

    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S_VAL",
        help="Global RNG seed. If None, random.",
    )
    common_group.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="CONFIG_FILE",
        help="Path to TOML configuration file. CLI args override config values.",
    )
    common_group.add_argument(
        "--base-output-dir",
        type=str,
        default="generated_output_dask",
        metavar="PATH",
        help="Base output directory.",
    )
    common_group.add_argument(
        "--data-subdir",
        type=str,
        default="data",
        metavar="SUB_D",
        help="Subdirectory for data.",
    )
    common_group.add_argument(
        "--plan-subdir",
        type=str,
        default="plans",
        metavar="SUB_P",
        help="Subdirectory for plans/DOTs.",
    )

    dask_group = parser.add_argument_group("Dask Parameters")
    dask_group.add_argument(
        "--dask-workers",
        type=int,
        default=None,
        help="Number of Dask workers. Default: Dask's choice or cpu_count // 2.",
    )
    dask_group.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=None,
        help="Threads per Dask worker. Default: Dask's choice or 2.",
    )
    dask_group.add_argument(
        "--dask-memory-limit",
        type=str,
        default="auto",
        help="Memory limit per Dask worker (e.g., '2GB', 'auto').",
    )
    dask_group.add_argument(
        "--dask-num-data-partitions-per-relation",
        type=int,
        default=0,
        help="Number of Dask partitions for each relation's unique tuple generation. Default (0) means Dask decides, or it's set to number of workers.",
    )

    log_group = parser.add_argument_group("Logging Parameters")
    log_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable detailed DEBUG logging."
    )
    log_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        metavar="LOG_P",
        help="Optional path for log file.",
    )

    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run selectivity analysis on generated plans and data.",
    )
    analysis_group.add_argument(
        "--run-on-the-fly-analysis",
        action="store_true",
        help="Run analysis for each plan as it's generated (creates individual analysis files).",
    )

    parsed_args = parser.parse_args()

    # Load config file if provided and merge with CLI args
    config = {}
    if parsed_args.config:
        config_path = pathlib.Path(parsed_args.config)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        config = load_config_file(config_path)

    # Merge config with CLI args (CLI takes precedence)
    merged_config = merge_config_with_args(config, parsed_args)
    global_args = create_global_args_from_merged(merged_config)

    setup_logging(global_args.verbose, global_args.log_file)
    logger.info(f"Using global seed: {global_args.seed}")

    # --- Argument Validation ---
    if global_args.duplication_factor < 1:
        parser.error("--duplication-factor must be >= 1.")
    if global_args.distribution == "zipf" and (
        global_args.skew is None or global_args.skew <= 1.0
    ):
        parser.error("--skew must be > 1.0 for Zipf distribution.")
    if not (0.0 <= global_args.null_percentage < 1.0):
        parser.error("--null-percentage must be in [0.0, 1.0).")
    if global_args.domain_size < 1:
        parser.error("--domain-size must be >= 1.")
    if global_args.relations < 0:
        parser.error("--relations cannot be negative.")
    if global_args.relations > 0 and global_args.attributes <= 0:
        parser.error("--attributes must be positive if relations > 0.")
    if global_args.unique_tuples < 0 and global_args.relations > 0:
        parser.error("--unique-tuples cannot be negative if relations > 0.")
    if global_args.plans < 0:
        parser.error("--plans cannot be negative.")

    data_output_path = global_args.base_output_dir / global_args.data_subdir_name
    plan_output_path = global_args.base_output_dir / global_args.plan_subdir_name

    try:
        data_output_path.mkdir(parents=True, exist_ok=True)
        if global_args.plans > 0 or global_args.generate_dot_visualization:
            plan_output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directories: {e}")
        parser.error(f"Could not create output directories: {e}")

    logger.info("Starting datagen.py with configuration:")
    for field_name, value in global_args.__dict__.items():
        logger.info(f"{INDENT}{field_name}: {value}")
    logger.info("-" * 30)

    client: Optional[Client] = None
    cluster: Optional[LocalCluster] = None
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
    except Exception as e:
        logger.error(
            f"Failed to start Dask client/cluster: {e}. This program requires Dask to function."
        )
        return

    relation_names: List[str] = []
    dask_data_tasks = []
    if global_args.relations > 0:
        if global_args.attributes <= 0:
            logger.error(
                f"Cannot generate relations with {global_args.attributes} attributes."
            )
        else:
            logger.info("Preparing Dask tasks for data generation...")
            relation_names = [f"R{i}" for i in range(global_args.relations)]
            for i, r_name in enumerate(relation_names):
                rel_seed = global_args.seed + i + 101

                output_base_name = r_name
                if global_args.data_output_format == "parquet":
                    # For Parquet, output_path_base is the directory name
                    output_file_or_dir_path = (
                        data_output_path / f"{output_base_name}.parquet"
                    )
                else:  # csv, json
                    output_file_or_dir_path = (
                        data_output_path
                        / f"{output_base_name}.{global_args.data_output_format}"
                    )

                num_partitions_for_relation = (
                    global_args.dask_num_data_partitions_per_relation
                )
                if (
                    num_partitions_for_relation <= 0
                ):  # Default to number of workers if not specified or invalid
                    num_partitions_for_relation = (
                        len(client.scheduler_info()["workers"])
                        if client
                        else cpu_count()
                    )

                spec = RelationGenSpec(
                    name=r_name,
                    num_attributes=global_args.attributes,
                    num_unique_tuples=global_args.unique_tuples,
                    duplication_factor=global_args.duplication_factor,
                    distribution_name=global_args.distribution,
                    skew=global_args.skew,
                    domain_size=global_args.domain_size,
                    null_percentage=global_args.null_percentage,
                    output_format_name=global_args.data_output_format,
                    output_path_base=output_file_or_dir_path,
                    seed=rel_seed,
                    num_data_partitions=num_partitions_for_relation,
                )
                dask_data_tasks.append(
                    dask.delayed(generate_one_relation_dask_task)(spec)
                )

            if dask_data_tasks:
                logger.info(
                    f"Computing {len(dask_data_tasks)} Dask data generation tasks..."
                )
                # Using dask.compute with progress for console feedback
                computed_results = dask.compute(*dask_data_tasks)
                progress(
                    computed_results
                )  # This will show progress if tasks are substantial

                # Check results after computation
                for res_path in computed_results:
                    if res_path:
                        logger.debug(f"Data task completed, output: {res_path}")
                    else:
                        logger.error(
                            "A data generation task may have failed (returned None). Check worker logs."
                        )
                logger.info("Data generation phase complete.")
    else:
        logger.info("No relations to generate (--relations is 0).")
    logger.info("-" * 30)

    fundamental_plan_id_counter: int = 0
    dask_plan_tasks = []
    if global_args.plans > 0:
        if not relation_names:
            logger.warning(
                "No relations were generated, cannot generate execution plans."
            )
        else:
            logger.info("Preparing Dask tasks for plan generation...")

            actual_max_rels_in_scope = global_args.relations
            if global_args.max_join_relations is not None:
                if global_args.max_join_relations < 0:
                    actual_max_rels_in_scope = 0
                else:
                    actual_max_rels_in_scope = min(
                        global_args.max_join_relations, global_args.relations
                    )

            global_plan_args_subset = GlobalCLIArgsPlanSubset(
                plan_granularity=global_args.plan_granularity,
                plan_output_format=global_args.plan_output_format,
                generate_dot_visualization=global_args.generate_dot_visualization,
                num_attributes_per_relation=global_args.attributes,
                run_on_the_fly_analysis=global_args.run_on_the_fly_analysis,
                data_output_path=data_output_path if global_args.run_on_the_fly_analysis else None,
                data_output_format=global_args.data_output_format,
                base_output_path=global_args.base_output_dir if global_args.run_on_the_fly_analysis else None,
            )

            for pattern_name in global_args.join_pattern:
                for _ in range(global_args.plans):
                    plan_seed = (
                        global_args.seed + fundamental_plan_id_counter * 1000 + 301
                    )

                    spec = FundamentalPlanSpecForWorker(
                        plan_id=fundamental_plan_id_counter,
                        all_relation_names=list(relation_names),
                        join_pattern_name=pattern_name,
                        max_relations_in_scope=actual_max_rels_in_scope,
                        plan_output_dir=plan_output_path,
                        seed_for_plan=plan_seed,
                    )
                    dask_plan_tasks.append(
                        dask.delayed(process_one_fundamental_plan_task)(
                            spec, global_plan_args_subset
                        )
                    )
                    fundamental_plan_id_counter += 1

            if dask_plan_tasks:
                logger.info(
                    f"Computing {len(dask_plan_tasks)} Dask plan generation tasks..."
                )
                plan_results = dask.compute(*dask_plan_tasks)
                progress(plan_results)
                logger.info("Execution plan generation phase complete.")

    elif global_args.generate_dot_visualization and global_args.plans == 0:
        logger.info("Skipping DOT visualization as --plans is 0.")
    else:
        logger.info("Skipping execution plan generation (--plans set to 0).")

    logger.info("-" * 30)

    # Run batch analysis if requested (and not already done on-the-fly)
    if global_args.run_analysis and not global_args.run_on_the_fly_analysis:
        if global_args.relations > 0 and global_args.plans > 0:
            logger.info("Running batch selectivity analysis on generated data and plans...")
            try:
                from analysis_module import analyze_all_plans
                analyses = analyze_all_plans(
                    plan_output_path,
                    data_output_path,
                    global_args.data_output_format,
                    global_args.base_output_dir
                )
                logger.info(f"Batch analysis complete. Processed {len(analyses)} plans.")
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
        else:
            logger.warning("Cannot run analysis: no data or plans were generated.")
    elif global_args.run_analysis and global_args.run_on_the_fly_analysis:
        logger.info("Analysis was performed on-the-fly during plan generation.")
        # Optionally aggregate individual analysis files
        if global_args.relations > 0 and global_args.plans > 0:
            try:
                from analysis_module import aggregate_individual_analyses
                logger.info("Aggregating individual analysis files...")
                aggregate_individual_analyses(global_args.base_output_dir, plan_output_path)
                logger.info("Analysis aggregation complete.")
            except Exception as e:
                logger.warning(f"Analysis aggregation failed: {e}")
    else:
        logger.info("Skipping analysis (--run-analysis not specified).")

    logger.info("-" * 30)

    if client:
        try:
            logger.info("Shutting down Dask client and cluster...")
            client.close()
            if cluster:
                cluster.close(timeout=10)  # Add timeout to cluster close
            logger.info("Dask client and cluster shut down.")
        except Exception as e:
            logger.warning(f"Error shutting down Dask client/cluster: {e}")

    logger.info("datagen.py finished.")


if __name__ == "__main__":
    run_datagen()
