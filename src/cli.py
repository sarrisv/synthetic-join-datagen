import argparse
import dataclasses
import logging
import pathlib
import random
import sys
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from plan_generation import FORMATTERS
from registries import DISTRIBUTIONS, JOIN_PATTERNS

logger = logging.getLogger("cli")


@dataclass
# Holds all configuration parameters for a generation run
class ArgsConfig:
    name: Optional[str] = None
    relations: int = 3
    attributes: int = 3
    unique_tuples: int = 1000
    duplication_factor: int = 1
    distribution: str = "normal"
    dist_skew: Optional[float] = None
    dist_std_dev: Optional[float] = None
    domain_size: int = 1000
    null_percentage: float = 0.0
    data_output_format: str = "csv"
    plans: int = 1
    join_pattern: List[str] = field(default_factory=lambda: ["random"])
    plan_granularity: List[str] = field(default_factory=lambda: ["table"])
    max_join_relations: Optional[int] = None
    plan_output_format: List[str] = field(default_factory=lambda: ["txt"])
    gen_dot_viz: bool = False
    seed: Optional[int] = None
    base_output_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("./generated_output")
    )
    data_subdir: str = "data"
    plan_subdir: str = "plans"
    analysis_subdir: str = "analysis"
    verbose: int = 0
    detailed_output: bool = False
    log_file: Optional[pathlib.Path] = None
    dask_workers: Optional[int] = None
    dask_threads_per_worker: Optional[int] = None
    dask_memory_limit: Optional[str] = "auto"
    dask_partitions_per_relation: int = 0
    analyze: bool = False


def _get_arg_defaults() -> Dict[str, Any]:
    # Introspects the ArgsConfig dataclass to get default values
    defaults = {}
    for f in dataclasses.fields(ArgsConfig):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def _add_data_group(parser: argparse.ArgumentParser) -> None:
    # Adds data generation arguments to the parser
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
        dest="unique_tuples",
        help="Unique tuples per relation.",
    )
    data_group.add_argument(
        "--duplication-factor",
        type=int,
        metavar="FACTOR",
        dest="duplication_factor",
        help="Duplication factor (>=1).",
    )
    data_group.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS.keys()),
        help="Distribution for non-PKs (non-primary key attributes).",
    )
    data_group.add_argument(
        "--dist-skew",
        type=float,
        metavar="SKEW",
        dest="dist_skew",
        help="Skew 'a' for ZipfDistribution (>1.0). Default: 2.0.",
    )
    data_group.add_argument(
        "--dist-std-dev",
        type=float,
        metavar="STDDEV",
        dest="dist_std_dev",
        help="Standard deviation for NormalDistribution. Default: domain_size / 5.",
    )
    data_group.add_argument(
        "--domain-size",
        type=int,
        metavar="VALUE",
        dest="domain_size",
        help="Maximum value for non-PK attributes in the domain [1, VALUE].",
    )
    data_group.add_argument(
        "--null-percentage",
        type=float,
        metavar="PERCENT",
        dest="null_percentage",
        help="Null percentage [0.0, 1.0) for non-PKs.",
    )
    data_group.add_argument(
        "--data-output-format",
        choices=["csv", "json", "parquet"],
        dest="data_output_format",
        help="Output format for generated data files.",
    )


def _add_plan_group(parser: argparse.ArgumentParser) -> None:
    # Adds plan generation arguments to the parser
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
        dest="join_pattern",
        help="Base join pattern/s",
    )
    plan_group.add_argument(
        "--plan-granularity",
        nargs="+",
        choices=["table", "attribute"],
        metavar="TYPE",
        dest="plan_granularity",
        help="Output plan granularity/ies.",
    )
    plan_group.add_argument(
        "--max-join-relations",
        type=int,
        metavar="COUNT",
        dest="max_join_relations",
        help="Max relations in plan scope",
    )
    plan_group.add_argument(
        "--plan-output-format",
        nargs="+",
        choices=list(FORMATTERS.keys()),
        metavar="FORMAT",
        dest="plan_output_format",
        help="Format(s) for derived plan files",
    )
    plan_group.add_argument(
        "--gen-dot-viz",
        action="store_true",
        dest="gen_dot_viz",
        help="Generate DOT visualization of base join patterns",
    )


def _add_common_group(parser: argparse.ArgumentParser) -> None:
    # Adds common arguments like seeding and output paths to the parser
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
        dest="config_file",
        help="Path to TOML config file. If used, no other configuration CLI arguments are allowed.",
    )
    common_group.add_argument(
        "--base-output-dir",
        type=str,
        metavar="PATH",
        dest="base_output_dir",
        help="Base output directory",
    )
    common_group.add_argument(
        "--data-subdir",
        type=str,
        metavar="NAME",
        dest="data_subdir",
        help="Subdirectory for data",
    )
    common_group.add_argument(
        "--plan-subdir",
        type=str,
        metavar="NAME",
        dest="plan_subdir",
        help="Subdirectory for plans/DOTs",
    )
    common_group.add_argument(
        "--analysis-subdir",
        type=str,
        metavar="NAME",
        dest="analysis_subdir",
        help="Subdirectory for analysis of plans",
    )


def _add_dask_group(parser: argparse.ArgumentParser) -> None:
    # Adds Dask-specific configuration arguments to the parser
    dask_group = parser.add_argument_group("Dask Parameters")
    dask_group.add_argument(
        "--dask-workers",
        type=int,
        dest="dask_workers",
        help="Number of Dask workers",
    )
    dask_group.add_argument(
        "--dask-threads-per-worker",
        type=int,
        dest="dask_threads_per_worker",
        help="Threads per Dask worker",
    )
    dask_group.add_argument(
        "--dask-memory-limit",
        type=str,
        dest="dask_memory_limit",
        help="Memory limit per Dask worker (e.g., '2GB', 'auto')",
    )
    dask_group.add_argument(
        "--dask-partitions-per-relation",
        type=int,
        dest="dask_partitions_per_relation",
        help="Number of Dask partitions for each relation's unique tuple generation",
    )


def _add_logging_group(parser: argparse.ArgumentParser) -> None:
    # Adds logging control arguments to the parser
    log_group = parser.add_argument_group("Logging Parameters")
    log_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Set level of verbosity",
    )
    log_group.add_argument(
        "-d",
        "--detailed-output",
        action="store_true",
        dest="detailed_output",
        help="Use more detailed logging format",
    )
    log_group.add_argument(
        "--log-file",
        type=pathlib.Path,
        metavar="PATH",
        dest="log_file",
        help="Optional path for log file",
    )


def _add_analysis_group(parser: argparse.ArgumentParser) -> None:
    # Adds analysis arguments to the parser
    analysis_group = parser.add_argument_group("Analysis Parameters")
    analysis_group.add_argument(
        "--analyze",
        action="store_true",
        help="Run selectivity analysis for each generated plan as it is created",
    )


def _setup_logging(
    verbosity: int, detailed: bool, log_file_path: Optional[pathlib.Path]
) -> None:
    # Configure the global logger for the application
    # Remove existing handlers to prevent duplicate output in case of re-initialization
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set logging level based on verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    if verbosity >= 2:
        log_level = logging.DEBUG
    elif verbosity == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.getLogger().setLevel(log_level)

    # Silence noisy Dask shuffle warnings unless in debug mode
    logging.getLogger("distributed.shuffle._scheduler_plugin").setLevel(
        logging.DEBUG if verbosity >= 2 else logging.ERROR
    )

    # Configure console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    if detailed:
        # Use a more detailed format if requested
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)-8s - %(name)-18s - %(message)s",
            datefmt="%H:%M:%S",
        )
        ch.setFormatter(console_formatter)
    logging.root.addHandler(ch)

    if log_file_path:
        # Configure file handler if a path is provided
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file_path, mode="w")
            fh.setLevel(logging.DEBUG)  # Always log debug level to file
            if detailed:
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)-18s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s",
                    datefmt="%H:%M:%S",
                )
                fh.setFormatter(file_formatter)
            logging.root.addHandler(fh)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file_path}: {e}")


def _validate_args(args: ArgsConfig) -> None:
    # Performs basic semantic validation on the configuration arguments
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


def _load_configs_from_file(cli_args: argparse.Namespace) -> List[ArgsConfig]:
    # Loads one or more run configurations from a TOML file
    config_path = pathlib.Path(cli_args.config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Parse the TOML file
    try:
        with config_path.open("rb") as f:
            toml_config = tomllib.load(f)
        logger.info(f"Loaded configuration from: {config_path}")
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Error decoding TOML file {config_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to load or process config file {config_path}: {e}")
        raise

    # Get default values to use as a base
    defaults = _get_arg_defaults()

    # Apply global settings from the [global] section, if it exists
    global_settings_raw = toml_config.get("global", {})
    global_settings = {k.replace("-", "_"): v for k, v in global_settings_raw.items()}

    # Allow overriding certain global settings from the CLI
    if cli_args.verbose > 0:
        global_settings["verbose"] = cli_args.verbose
    if cli_args.detailed_output:
        global_settings["detailed_output"] = cli_args.detailed_output
    if cli_args.log_file is not None:
        global_settings["log_file"] = cli_args.log_file

    # Check for an 'iterations' table to support multiple runs
    iterations_data = toml_config.get("iterations", [])
    if not iterations_data:
        # If no 'iterations', treat the file as a single configuration
        single_run_data_raw = {}
        for section_name, section_content in toml_config.items():
            # Collect all top-level tables that are not 'global'
            if section_name != "global" and isinstance(section_content, dict):
                single_run_data_raw.update(section_content)

        single_run_data = {
            k.replace("-", "_"): v for k, v in single_run_data_raw.items()
        }
        # Merge defaults, global settings, and the single run's settings
        final_config_dict = {**defaults, **global_settings, **single_run_data}
        # Ensure path-like strings are converted to Path objects
        if "base_output_dir" in final_config_dict and isinstance(
            final_config_dict["base_output_dir"], str
        ):
            final_config_dict["base_output_dir"] = pathlib.Path(
                final_config_dict["base_output_dir"]
            )
        return [ArgsConfig(**final_config_dict)]  # Return as a list with one item

    configs = []
    # Process each item in the 'iterations' list as a separate run
    for i, iter_data_raw in enumerate(iterations_data):
        iter_data = {k.replace("-", "_"): v for k, v in iter_data_raw.items()}
        # Assign a default name if one isn't provided
        if "name" not in iter_data:
            iter_data["name"] = f"iteration_{i}"

        # Merge defaults, global settings, and this iteration's settings
        final_config_dict = {**defaults, **global_settings, **iter_data}
        if "base_output_dir" in final_config_dict and isinstance(
            final_config_dict["base_output_dir"], str
        ):
            final_config_dict["base_output_dir"] = pathlib.Path(
                final_config_dict["base_output_dir"]
            )
        configs.append(ArgsConfig(**final_config_dict))
    return configs


def load_config() -> List[ArgsConfig]:
    # Main function to parse CLI args and load configurations
    parser = argparse.ArgumentParser(
        description="Dataset and Execution Plan Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    _add_data_group(parser)
    _add_plan_group(parser)
    _add_common_group(parser)
    _add_dask_group(parser)
    _add_logging_group(parser)
    _add_analysis_group(parser)

    # Set defaults from the dataclass before parsing
    defaults = _get_arg_defaults()
    parser.set_defaults(**defaults)
    cli_args = parser.parse_args()

    if cli_args.config_file:
        # When --config-file is used, most other CLI args are disallowed for clarity
        user_set_args = {
            key for key, val in vars(cli_args).items() if val != defaults.get(key)
        }
        # Define the few flags that are allowed to be used with a config file
        allowed_control_flags = {
            "config_file",
            "verbose",
            "log_file",
            "detailed_output",
        }
        conflicting_args = user_set_args - allowed_control_flags  # Find any conflicts

        if conflicting_args:
            # To provide a helpful error, find the original flag name (e.g., '--max-relations')
            conflicting_arg_dest = conflicting_args.pop()
            offending_flag = next(
                (
                    option
                    for action in parser._actions
                    if action.dest == conflicting_arg_dest
                    for option in action.option_strings
                ),
                f"--{conflicting_arg_dest.replace('_', '-')}",
            )
            parser.error(
                f"argument {offending_flag} cannot be used with --config-file.\n\n"
                "Please use either a configuration file OR command-line arguments for settings, but not both.\n"
                "The only flags allowed with --config-file are --verbose (-v), --log-file, and --detailed-output (-d)."
            )

    configs: List[ArgsConfig]

    if cli_args.config_file:
        try:
            configs = _load_configs_from_file(cli_args)
        except (FileNotFoundError, tomllib.TOMLDecodeError, Exception) as e:
            # Errors during file loading are fatal
            parser.error(str(e))
    else:
        # If no config file, create a single configuration from the parsed CLI args
        configs = [ArgsConfig(**vars(cli_args))]

    # Handle seeding logic for reproducibility
    seed = None
    if configs and configs[0].seed is not None:
        # A seed in the first config (or global) acts as a base seed
        seed = configs[0].seed

    for i, config in enumerate(configs):
        if config.seed is None:
            if seed is not None:
                # If a base seed was set, derive deterministic seeds for other runs
                config.seed = seed + i * 10000  # Use a large step for variety
            else:
                # Assign a random seed if no base seed is provided
                config.seed = random.randint(1, 2**32 - 1)
        if i == 0 and seed is None:
            seed = config.seed  # The first config's seed becomes the base
        _validate_args(config)

    # Setup logging based on the first configuration (or CLI args if no file)
    if configs:
        _setup_logging(
            configs[0].verbose, configs[0].detailed_output, configs[0].log_file
        )

    return configs
