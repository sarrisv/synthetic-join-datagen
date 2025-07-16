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
class AppConfig:
    relations: int = 3
    attributes: int = 3
    unique_tuples: int = 1000
    duplication_factor: int = 1
    distribution: str = "normal"
    dist_skew: Optional[float] = None
    dist_mean: Optional[float] = None
    dist_std_dev: Optional[float] = None
    dist_clip_mode: str = "center"
    domain_size: int = 1000
    null_percentage: float = 0.0
    data_output_format: str = "csv"
    plans: int = 1
    join_pattern: List[str] = field(default_factory=lambda: ["random"])
    plan_granularity: List[str] = field(default_factory=lambda: ["table"])
    max_join_relations: Optional[int] = None
    plan_output_format: List[str] = field(default_factory=lambda: ["txt"])
    generate_dot_visualization: bool = False
    seed: Optional[int] = None
    base_output_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("../generated_output")
    )
    data_subdir: str = "data"
    plan_subdir: str = "plans"
    analysis_subdir: str = "analysis"
    verbose: int = 0
    log_file: Optional[pathlib.Path] = None
    dask_workers: Optional[int] = None
    dask_threads_per_worker: Optional[int] = None
    dask_memory_limit: Optional[str] = "auto"
    dask_partitions_per_relation: int = 0
    run_on_the_fly_analysis: bool = False


def get_arg_defaults() -> Dict[str, Any]:
    defaults = {}
    for f in dataclasses.fields(AppConfig):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def add_data_group(parser: argparse.ArgumentParser) -> None:
    data_group = parser.add_argument_group("Data Generation Parameters")
    data_group.add_argument(
        "--relations", type=int, metavar="COUNT", help="Number of relations."
    )
    data_group.add_argument(
        "--attributes",
        type=int,
        metavar="COUNT",
        help="Attributes per relation (attr0 is PK).",
    )
    data_group.add_argument(
        "--unique-tuples",
        type=int,
        metavar="COUNT",
        help="Unique tuples per relation.",
    )
    data_group.add_argument(
        "--duplication-factor",
        dest="duplication_factor",
        type=int,
        metavar="FACTOR",
        help="Duplication factor (>=1).",
    )
    data_group.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS.keys()),
        help="Distribution for non-PKs (non-primary key attributes).",
    )
    data_group.add_argument(
        "--dist-skew",
        dest="dist_skew",
        type=float,
        metavar="SKEW",
        help="Skew 'a' for ZipfDistribution (>1.0). Default: 2.0.",
    )
    data_group.add_argument(
        "--dist-mean",
        dest="dist_mean",
        type=float,
        metavar="MEAN",
        help="Mean for NormalDistribution. Default: domain_size / 2.",
    )
    data_group.add_argument(
        "--dist-std-dev",
        dest="dist_std_dev",
        type=float,
        metavar="STDDEV",
        help="Standard deviation for NormalDistribution. Default: domain_size / 5.",
    )
    data_group.add_argument(
        "--dist-clip-mode",
        dest="dist_clip_mode",
        choices=["center", "range"],
        help="Clipping mode for NormalDistribution. 'center' clips around the mean, 'range' clips to [1, domain_size]. Default: 'center'.",
    )
    data_group.add_argument(
        "--domain-size",
        type=int,
        metavar="VALUE",
        help="Maximum value for non-PK attributes in the domain [1, VALUE].",
    )
    data_group.add_argument(
        "--null-percentage",
        dest="null_percentage",
        type=float,
        metavar="PERCENT",
        help="Null percentage [0.0, 1.0) for non-PKs.",
    )
    data_group.add_argument(
        "--data-output-format",
        choices=["csv", "json", "parquet"],
        help="Output format for generated data files.",
    )


def add_plan_group(parser: argparse.ArgumentParser) -> None:
    plan_group = parser.add_argument_group("Execution Plan Generation Parameters")
    plan_group.add_argument(
        "--plans",
        type=int,
        metavar="COUNT",
        help="Base plans per join pattern",
    )
    plan_group.add_argument(
        "--join-pattern",
        nargs="+",
        choices=list(JOIN_PATTERNS.keys()),
        metavar="PATTERN",
        help="Base join pattern/s",
    )
    plan_group.add_argument(
        "--plan-granularity",
        dest="plan_granularity",
        nargs="+",
        choices=["table", "attribute"],
        metavar="TYPE",
        help="Output plan granularity/ies.",
    )
    plan_group.add_argument(
        "--max-join-relations",
        type=int,
        metavar="COUNT",
        help="Max relations in plan scope",
    )
    plan_group.add_argument(
        "--plan-output-format",
        nargs="+",
        choices=list(FORMATTERS.keys()),
        metavar="FORMAT",
        help="Format(s) for derived plan files",
    )
    plan_group.add_argument(
        "--generate-dot-visualization",
        action="store_true",
        help="Generate DOT viz of base joins",
    )


def add_common_groups(parser: argparse.ArgumentParser) -> None:
    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument(
        "--seed",
        type=int,
        metavar="SEED_VAL",
        help="Global RNG seed",
    )
    common_group.add_argument(
        "--config-file",
        type=str,
        metavar="PATH",
        help="Path to TOML config file. If used, no other configuration CLI arguments are allowed.",
    )
    common_group.add_argument(
        "--base-output-dir",
        dest="base_output_dir",
        type=str,
        metavar="PATH",
        help="Base output directory",
    )
    common_group.add_argument(
        "--data-subdir",
        dest="data_subdir",
        type=str,
        metavar="NAME",
        help="Subdirectory for data",
    )
    common_group.add_argument(
        "--plan-subdir",
        dest="plan_subdir",
        type=str,
        metavar="NAME",
        help="Subdirectory for plans/DOTs",
    )
    common_group.add_argument(
        "--analysis-subdir",
        dest="analysis_subdir",
        type=str,
        metavar="NAME",
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
        "--dask-partitions-per-relation",
        dest="dask_partitions_per_relation",
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
        metavar="PATH",
        help="Optional path for log file",
    )


def add_analysis_groups(parser: argparse.ArgumentParser) -> None:
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--run-on-the-fly-analysis",
        action="store_true",
        help="Run analysis for each plan as it's generated (creates individual analysis files)",
    )


def load_config_file(config_path: pathlib.Path) -> Dict[str, Any]:
    """Loads a TOML config file, flattens it, and prepares for dataclass instantiation."""
    try:
        with config_path.open("rb") as f:
            grouped_config = tomllib.load(f)
        logger.info(f"Loaded configuration from: {config_path}")

        # Flatten the nested TOML structure and normalize keys
        config = {
            key.replace("-", "_"): value
            for section in grouped_config.values()
            for key, value in section.items()
        }

        # Ensure Path objects are created for path-like strings
        if "base_output_dir" in config and isinstance(config["base_output_dir"], str):
            config["base_output_dir"] = pathlib.Path(config["base_output_dir"])
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error decoding TOML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load or process config file {config_path}: {e}")
        raise


def setup_logging(verbosity: int, log_file_path: Optional[pathlib.Path]) -> None:
    # Remove existing handlers to prevent duplicate output if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set logging level
    if verbosity >= 1:
        log_level = logging.DEBUG
    else:  # verbosity == 0
        log_level = logging.INFO
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


def validate_args(args: AppConfig) -> None:
    errors = []

    if args.duplication_factor < 1:
        errors.append(
            f"--duplication-factor must be >= 1 (got {args.duplication_factor})."
        )
    if (
        args.distribution == "zipf"
        and args.dist_skew is not None
        and args.dist_skew <= 1.0
    ):
        errors.append(
            f"--dist-skew must be > 1.0 for ZipfDistribution (got {args.dist_skew})."
        )
    if not (0.0 <= args.null_percentage < 1.0):
        errors.append(
            f"--null-percentage must be in [0.0, 1.0) (got {args.null_percentage})."
        )
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


class CustomHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """Allows for both default values and raw text formatting in help messages."""

    pass


def load_config() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Dataset and Execution Plan Generator",
        formatter_class=CustomHelpFormatter,
    )

    add_data_group(parser)
    add_plan_group(parser)
    add_common_groups(parser)
    add_dask_group(parser)
    add_logging_group(parser)
    add_analysis_groups(parser)

    argv = sys.argv[1:]
    using_config_file = any(arg.startswith("--config-file") for arg in argv)

    if using_config_file:
        # Enforce that if --config-file is used, no other configuration arguments
        # are allowed on the command line, except for control-flow flags like
        # verbosity. This is done by manually inspecting sys.argv, as argparse
        # doesn't easily expose which arguments were user-provided vs. defaults.
        # Dynamically build a set of all configuration-related option strings (flags)
        config_flags = set()
        # These "control" flags are always allowed, even with a config file.
        allowed_dests = {"config_file", "verbose", "log_file"}

        for action in parser._actions:
            if action.dest not in allowed_dests:
                config_flags.update(action.option_strings)

        found_conflicting_arg = None
        for arg in argv:
            # Match exact flags (e.g., '--seed') or flags with values ('--seed=123')
            if any(arg == flag or arg.startswith(f"{flag}=") for flag in config_flags):
                found_conflicting_arg = arg
                break

        if found_conflicting_arg:
            parser.error(
                f"argument {found_conflicting_arg} cannot be used with --config-file.\n\n"
                "Please use either a configuration file OR command-line arguments for settings, but not both.\n"
                "The only flags allowed with --config-file are --verbose (-v) and --log-file."
            )

    # --- If we passed the check, or if not using a config file, proceed with normal parsing ---
    parser.set_defaults(**get_arg_defaults())
    cli_args = parser.parse_args()
    config: AppConfig

    if cli_args.config_file:
        # --- Config File Mode ---
        config_path = pathlib.Path(cli_args.config_file)
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}")

        config_from_file = load_config_file(config_path)

        # Start with dataclass defaults and overlay with values from TOML file
        final_config = get_arg_defaults()
        if config_from_file:
            final_config.update(config_from_file)

        # Manually apply the allowed CLI args (e.g., logging) over the top
        final_config["verbose"] = cli_args.verbose
        final_config["log_file"] = cli_args.log_file

        config = AppConfig(**final_config)
    else:
        # --- CLI Arguments Mode ---
        config = AppConfig(**vars(cli_args))

    # Common post-processing for both modes
    if config.seed is None:
        config.seed = random.randint(1, 2**32 - 1)

    validate_args(config)
    setup_logging(config.verbose, config.log_file)

    return config
