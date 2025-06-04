import argparse
import dataclasses
import logging
import pathlib
import random
import sys
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from registries import DISTRIBUTIONS, JOIN_PATTERNS, FORMATTERS

logger = logging.getLogger(__name__)


@dataclass
class GlobalCLIArgs:
    # Data Generation Parameters
    relations: int = 3
    attributes: int = 3
    unique_tuples: int = 1000
    dup_factor: int = 1
    distribution: str = "normal"
    skew: Optional[float] = 2
    domain_size: int = 1000
    null_pct: float = 0.0
    data_output_format: str = "csv"

    # Execution Plan Generation Parameters
    plans: int = 1
    join_pattern: List[str] = field(default_factory=lambda: ["random"])
    plan_granularity: List[str] = field(default_factory=lambda: ["table"])
    max_join_relations: Optional[int] = None
    plan_output_format: List[str] = field(default_factory=lambda: ["txt"])
    generate_dot_visualization: bool = False

    # Common Parameters
    seed: Optional[int] = None
    base_output_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("generated_output")
    )
    data_subdir_name: str = "data"
    plan_subdir_name: str = "plans"
    analysis_subdir_name: str = "analysis"

    # Logging Parameters
    verbose: int = 0
    log_file: Optional[pathlib.Path] = None

    # Dask Parameters
    dask_workers: Optional[int] = None
    dask_threads_per_worker: Optional[int] = None
    dask_memory_limit: Optional[str] = "auto"
    dask_partitions_per_relation: int = 0

    # Analysis Parameters
    run_analysis: bool = False
    run_on_the_fly_analysis: bool = False


def get_arg_defaults() -> Dict[str, Any]:
    defaults = {}
    for f in dataclasses.fields(GlobalCLIArgs):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def add_data_group(parser: argparse.ArgumentParser) -> None:
    data_group = parser.add_argument_group("Data Generation Parameters")
    data_group.add_argument(
        "--relations", type=int, metavar="N", help="Number of relations."
    )
    data_group.add_argument(
        "--attributes",
        type=int,
        metavar="M",
        help="Attributes per relation (attr0 is PK).",
    )
    data_group.add_argument(
        "--unique-tuples",
        type=int,
        metavar="U",
        help="Unique tuples per relation.",
    )
    data_group.add_argument(
        "--duplication-factor",
        type=int,
        metavar="D",
        help="Duplication factor (>=1).",
    )
    data_group.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS.keys()),
        help="Distribution for non-PKs.",
    )
    data_group.add_argument(
        "--skew",
        type=float,
        metavar="S",
        help="Skew 'a' for Zipf (>1.0). Default: 2.0 if Zipf.",
    )
    data_group.add_argument(
        "--domain-size",
        type=int,
        metavar="MAX_V",
        help="Max value for non-PKs [1, MAX_V].",
    )
    data_group.add_argument(
        "--null-percentage",
        type=float,
        metavar="P_N",
        help="Null percentage [0.0, 1.0) for non-PKs.",
    )
    data_group.add_argument(
        "--data-output-format",
        choices=["csv", "json", "parquet"],
        help="Format for data files.",
    )


def add_plan_group(parser: argparse.ArgumentParser) -> None:
    plan_group = parser.add_argument_group("Execution Plan Generation Parameters")
    plan_group.add_argument(
        "--plans",
        type=int,
        metavar="N_P_PAT",
        help="Fundamental plans per join pattern",
    )
    plan_group.add_argument(
        "--join-pattern",
        nargs="+",
        choices=list(JOIN_PATTERNS.keys()),
        metavar="PAT",
        help="Fundamental join pattern/s",
    )
    plan_group.add_argument(
        "--plan-granularity",
        nargs="+",
        choices=["table", "attribute"],
        metavar="GRAN",
        help="Output plan granularity/ies.",
    )
    plan_group.add_argument(
        "--max-join-relations",
        type=int,
        metavar="MAX_R",
        help="Max relations in plan scope",
    )
    plan_group.add_argument(
        "--plan-output-format",
        nargs="+",
        choices=list(FORMATTERS.keys()),
        metavar="PLAN_FMT",
        help="Format(s) for derived plan files",
    )
    plan_group.add_argument(
        "--generate-dot-visualization",
        action="store_true",
        help="Generate DOT viz of fundamental joins",
    )


def add_common_groups(parser: argparse.ArgumentParser) -> None:
    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument(
        "--seed",
        type=int,
        metavar="S_VAL",
        help="Global RNG seed",
    )
    common_group.add_argument(
        "--config-file",
        type=str,
        metavar="CONFIG_FILE",
        help="Path to TOML configuration file, CLI args override config values",
    )
    common_group.add_argument(
        "--base-output-dir",
        type=str,
        metavar="PATH",
        help="Base output directory",
    )
    common_group.add_argument(
        "--data-subdir",
        type=str,
        metavar="SUB_D",
        help="Subdirectory for data",
    )
    common_group.add_argument(
        "--plan-subdir",
        type=str,
        metavar="SUB_P",
        help="Subdirectory for plans/DOTs",
    )
    common_group.add_argument(
        "--analysis-subdir",
        type=str,
        metavar="SUB_A",
        help="Subdirectory for analysis of plans",
    )


def add_dask_group(parser: argparse.ArgumentParser) -> None:
    dask_group = parser.add_argument_group("Dask Parameters")
    dask_group.add_argument(
        "--dask-workers",
        type=int,
        help="Number of Dask workers",
    )
    dask_group.add_argument(
        "--dask-threads-per-worker",
        type=int,
        help="Threads per Dask worker",
    )
    dask_group.add_argument(
        "--dask-memory-limit",
        type=str,
        help="Memory limit per Dask worker (e.g., '2GB', 'auto')",
    )
    dask_group.add_argument(
        "--dask-num-data-partitions-per-relation",
        type=int,
        help="Number of Dask partitions for each relation's unique tuple generation",
    )


def add_logging_group(parser: argparse.ArgumentParser) -> None:
    log_group = parser.add_argument_group("Logging Parameters")
    log_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Set level of verbosity",
    )
    log_group.add_argument(
        "--log-file",
        type=pathlib.Path,
        metavar="LOG_P",
        help="Optional path for log file",
    )


def add_analysis_groups(parser: argparse.ArgumentParser) -> None:
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run selectivity analysis on generated plans and data",
    )
    analysis_group.add_argument(
        "--run-on-the-fly-analysis",
        action="store_true",
        help="Run analysis for each plan as it's generated (creates individual analysis files)",
    )


def load_config_file(config_path: pathlib.Path) -> Dict[str, Any] | None:
    if config_path:
        config_path = pathlib.Path(config_path)
    try:
        with open(config_path, "rb") as f:
            grouped_config = tomllib.load(f)
        logger.info(f"Loaded configuration from: {config_path}")

        config = {}
        if grouped_config:
            for _, args_dict in grouped_config.items():
                for k, v in args_dict.items():
                    config[str(k).replace("-", "_")] = v

        if "base_output_dir" in config:
            config["base_output_dir"] = pathlib.Path(config["base_output_dir"])
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise


def setup_logging(verbosity: int, log_file_path: Optional[pathlib.Path]) -> None:
    # Remove existing handlers to prevent duplicate output if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set logging level
    log_level = logging.NOTSET
    if verbosity == 0:
        log_level = logging.INFO
    elif verbosity > 1:
        log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)

    # Setup logging to console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)-8s - %(name)-35s - %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(console_formatter)
    logging.getLogger().addHandler(ch)

    # Set up logging to log file, if set
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


def validate_args(args: GlobalCLIArgs) -> None:
    errors = []

    if args.dup_factor < 1:
        errors.append(f"--duplication-factor must be >= 1 (got {args.dup_factor}).")
    if args.distribution == "zipf" and (args.skew is None or args.skew <= 1.0):
        errors.append(f"--skew must be > 1.0 for Zipf distribution (got {args.skew}).")
    if not (0.0 <= args.null_pct < 1.0):
        errors.append(f"--null-percentage must be in [0.0, 1.0) (got {args.null_pct}).")
    if args.domain_size < 1:
        errors.append(f"--domain-size must be positive (got {args.domain_size}).")
    if args.relations < 1:
        errors.append(f"--relations must be >= 1 (got {args.relations}).")
    if args.attributes < 1:
        errors.append(f"--attributes must be >= 1 (got {args.attributes}).")
    if args.unique_tuples < 1:
        errors.append(f"--unique-tuples must be >= 1 (got {args.unique_tuples}).")
    if args.plans < 0:
        errors.append(f"--plans cannot be negative (got {args.plans}).")

    if errors:
        for error_msg in errors:
            logger.error(f"Validation Error: {error_msg}")
        sys.exit(1)


def load_config() -> GlobalCLIArgs:
    parser = argparse.ArgumentParser(
        description="Dataset and Execution Plan Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add all the arguments, having it all here was too much
    add_data_group(parser)
    add_plan_group(parser)
    add_common_groups(parser)
    add_dask_group(parser)
    add_logging_group(parser)
    add_analysis_groups(parser)

    # Set defaults in the help message based on GlobalCLIArgs
    parser.set_defaults(**get_arg_defaults())

    # Parse any args provided via CLI, empty args are set to default value
    cli_args = parser.parse_args()

    # Load config file, if provided
    config_file_args = {}
    if cli_args.config_file:
        config_path = pathlib.Path(cli_args.config_file)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")
        config_file_args = load_config_file(config_path)

    # Merge CLI and config file args (config file takes precedence)
    merged_args = {}
    for f in dataclasses.fields(GlobalCLIArgs):
        if f.name in config_file_args:
            merged_args[f.name] = config_file_args[f.name]
        else:
            merged_args[f.name] = vars(cli_args)[f.name]

    if merged_args["seed"] is None:
        merged_args["seed"] = random.randint(1, 2**32 - 1)

    # Create instance of CLI arguments w/ parsed args
    global_args = GlobalCLIArgs(**merged_args)
    validate_args(global_args)

    setup_logging(global_args.verbose, global_args.log_file)

    return global_args
