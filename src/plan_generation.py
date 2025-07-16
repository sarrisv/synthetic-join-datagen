import json
import logging
import pathlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union, cast, Set, Type

import numpy as np

from analysis_module import analyze_single_plan, save_plan_analysis
from registries import JOIN_PATTERNS
from plan_structures import Plan, BinaryJoinStage, MultiwayJoinStage

logger = logging.getLogger("plangen")


# --- Plan Formatters ---
class Formatter(ABC):
# Tightly coupled with Plan object and used only for writing plan outputs
    """Abstract base class for formatting a derived plan into a specific output format."""

    @abstractmethod
    def format(self, derived_plan: Plan) -> Union[str, Dict[str, Any]]:
        """Formats the derived plan into either a string or a serializable dictionary."""
        pass


class TextFormatter(Formatter):
    """Formats the plan into a human-readable text file."""

    def format(self, derived_plan: Plan) -> str:
        return derived_plan.to_text()


class JSONFormatter(Formatter):
    """Formats the plan into a dictionary for JSON serialization."""

    def format(self, derived_plan: Plan) -> Dict[str, Any]:
        return derived_plan.to_dict()


# Maps file extensions to their corresponding formatter classes
FORMATTERS: Dict[str, Type[Formatter]] = {
    "txt": TextFormatter,
    "json": JSONFormatter,
}


@dataclass
# Holds global configuration settings for plan generation
class PlanConfig:
    """Relevant global CLI arguments for plan generation tasks"""

    plan_granularity: List[str]
    plan_output_format: List[str]
    gen_dot_viz: bool
    num_attrs: int
    analyze: bool
    data_output_path: Optional[pathlib.Path]
    base_output_path: Optional[pathlib.Path]
    analysis_subdir: str


@dataclass
# Defines the specification for a single base plan to be generated
class PlanSpec:
    """Specification for a base plan"""

    plan_id: int
    relations: List[str]
    pattern: str
    max_relations: int
    output_dir: pathlib.Path
    seed: int


def _derive_stages(
    plan_id: int,
    relations: List[str],
    pattern: str,
    joins: List[Tuple[str, str, int]],
    granularity: str,
) -> Plan:
    """
    Derives staged plan data (BinaryJoinStage or MultiwayJoinStage objects)
    based on granularity from the base join
    """
    if not joins:
        # Handle cases with no joins, returning a plan with empty stages
        return Plan(
            plan_id=plan_id,
            pattern=pattern,
            granularity=granularity,
            relations=relations,
            stages=[],
        )

    stages: Union[List[BinaryJoinStage], List[MultiwayJoinStage]]
    if granularity == "table":
        # 'table' granularity creates one join stage per pair of tables
        table_stages: List[BinaryJoinStage] = []
        # Aggregate all join attributes for each pair of relations
        aggregated_joins: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for r1, r2, attr_idx in joins:
            aggregated_joins[(r1, r2)].add(f"attr{attr_idx}")

        # Create a BinaryJoinStage for each unique pair of relations
        for pair, attrs_set in sorted(aggregated_joins.items()):
            table_stages.append(
                BinaryJoinStage(
                    rel1=pair[0], rel2=pair[1], join_attributes=sorted(list(attrs_set))
                )
            )
        stages = table_stages

    elif granularity == "attribute":
        # 'attribute' granularity creates one multi-way join stage per attribute
        attr_stages: List[MultiwayJoinStage] = []
        # Group all relations that are joined on the same attribute
        relations_by_attr: Dict[str, Set[str]] = defaultdict(set)
        for r1, r2, attr_idx in joins:
            attr_name = f"attr{attr_idx}"
            relations_by_attr[attr_name].add(r1)
            relations_by_attr[attr_name].add(r2)

        # Create a MultiwayJoinStage for each joining attribute
        for attr_name, relations_set in sorted(relations_by_attr.items()):
            attr_stages.append(
                MultiwayJoinStage(
                    joining_attribute=attr_name,
                    relations=sorted(list(relations_set)),
                )
            )
        stages = attr_stages
    else:
        # Handle unknown granularity, resulting in an empty stage list
        stages = []
        logger.error(f"Unknown granularity when deriving stages: {granularity}")

    # Construct the final Plan object with the derived stages
    return Plan(
        plan_id=plan_id,
        pattern=pattern,
        granularity=granularity,
        relations=relations,
        stages=stages,
    )


def _generate_dot(
    plan_id: int,
    pattern: str,
    relations: List[str],
    joins: List[Tuple[str, str, int]],
    plan_dir_path: pathlib.Path,
) -> None:
    """Generates a DOT file for the given base joins"""
    # Define the output file path for the DOT visualization
    file_path = plan_dir_path / f"plan_{plan_id}_{pattern}_base_joins_viz.dot"
    # Create a specific logger for this task
    task_logger = logging.getLogger(f"{__name__}.dot_viz.{plan_id}.{pattern}")
    task_logger.info(f"Generating DOT visualization: {file_path.name}")

    # Start building the DOT file content as a list of strings
    dot_lines: List[str] = [
        f"graph BaseJoins_{plan_id}_{pattern} {{",
        "  graph [",
        '    layout="circo"; overlap=false; center=true;',
        f'    label="Base Joins Visualization\\nPlan ID: {plan_id}\\nPattern: {pattern}\\nScope: {", ".join(relations)}";',
        "    labelloc=t; fontsize=10;",
        "  ]\n",
        "  node [shape=ellipse, fontsize=10];",
        "  edge [fontsize=8];",
    ]

    if not relations:
        dot_lines.append("\n  // No relations in scope for visualization")
    else:
        # Add all relations in the scope as nodes
        dot_lines.append("")
        for rel_name in relations:
            dot_lines.append(f'  "{rel_name}";')

        if not joins:
            dot_lines.append("\n  // No base joins to visualize")
        else:
            # Aggregate join attributes for each edge (pair of relations)
            edge_labels: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            for r1, r2, attr_idx in joins:
                attr_name = f"attr{attr_idx}"
                edge_labels[(r1, r2)].add(attr_name)

            # Add an edge for each join with its attributes as a label
            dot_lines.append("")
            for pair, attrs_set in sorted(edge_labels.items()):
                attrs_list = sorted(list(attrs_set))
                dot_lines.append(
                    f'  "{pair[0]}" -- "{pair[1]}" [label="{", ".join(attrs_list)}"];'
                )

    dot_lines.append("}")
    # Write the generated DOT content to the file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(dot_lines))
    except IOError as e:
        task_logger.error(f"Could not write DOT visualization file {file_path}: {e}")


def _write_formatted_plan(
    file_path: pathlib.Path,
    format_name: str,
    formatted_content: Union[str, Dict[str, Any]],
    task_logger: logging.Logger,
) -> None:
    """Writes formatted plan content to a file, handling different formats"""
    task_logger.debug(f"Writing plan to {file_path.name}")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if format_name == "json":
                # For JSON, dump the dictionary with indentation
                json.dump(formatted_content, f, indent=2)
            elif format_name == "txt":
                # For text, write the string content directly
                f.write(cast(str, formatted_content) + "\n")
    except IOError as e:
        task_logger.error(f"Could not write plan file {file_path}: {e}")


def _run_plan_analysis(
    spec: PlanSpec,
    plan_config: PlanConfig,
    granularity: str,
    task_logger: logging.Logger,
) -> None:
    """Runs and saves the analysis for a specific, newly generated plan."""
    # Construct the path to the text plan file, required for analysis
    plan_txt_path = (
        spec.output_dir / f"plan_{spec.plan_id}_{spec.pattern}_{granularity}.txt"
    )

    # If the required plan file doesn't exist, skip analysis
    if not plan_txt_path.exists():
        task_logger.warning(
            f"Plan file {plan_txt_path.name} not found, skipping its analysis."
        )
        return

    task_logger.debug(f"Running analysis for {plan_txt_path.name}")
    try:
        # Perform the actual analysis on the plan file and data
        analysis = analyze_single_plan(
            plan_txt_path, cast(pathlib.Path, plan_config.data_output_path)
        )
        # Determine the directory where analysis results should be saved
        analysis_output_dir = (
            cast(pathlib.Path, plan_config.base_output_path)
            / plan_config.analysis_subdir
        )
        # Define the full path for the individual analysis JSON file
        individual_json_path = (
            analysis_output_dir
            / f"analysis_plan_{spec.plan_id}_{spec.pattern}_{granularity}.json"
        )

        # Save the analysis results to the JSON file
        save_plan_analysis(analysis, individual_json_path)
        task_logger.info(f"Analysis JSON written to: {individual_json_path.name}")
    except Exception as e:
        # Catch any errors during the analysis process
        task_logger.warning(
            f"Analysis failed for plan {spec.plan_id}, granularity '{granularity}': {e}"
        )


def generate_plan(
    spec: PlanSpec, plan_config: PlanConfig, analysis_deps: Optional[List] = None
) -> None:
    """Dask task to generate one base plan and all its derived outputs (different granularities and formats)"""
    # Create a logger specific to this plan generation task
    task_logger = logging.getLogger(
        f"{__name__}.plan_worker.{spec.plan_id}.{spec.pattern}"
    )
    # The analysis_deps argument is not used directly in this function
    # Its purpose is to create a dependency in the Dask graph, ensuring data generation is complete
    # By accepting it as an argument, we ensure Dask won't start this task until
    # all tasks in analysis_deps (i.e, data and metadata generation) have completed
    # This is crucial for the plan analysis step.
    task_logger.info(
        f"Processing base plan ID {spec.plan_id} with pattern '{spec.pattern}'"
    )
    # Initialize a random number generator with a specific seed for reproducibility
    rng = np.random.default_rng(spec.seed)

    # --- 1. Select relations for the plan scope ---
    plan_scope: List[str] = []
    if not spec.relations:
        # Warn if no relations are provided to choose from
        if spec.max_relations > 0:
            task_logger.warning(
                f"Plan ID {spec.plan_id}: No global relations available to select scope from."
            )
    else:
        # Determine how many relations to select for this plan's scope
        num_relations_to_select = min(spec.max_relations, len(spec.relations))
        if num_relations_to_select > 0:
            # Randomly select a subset of relations without replacement
            plan_scope = sorted(
                rng.choice(
                    spec.relations, size=num_relations_to_select, replace=False
                ).tolist()
            )
        else:  # This happens if spec.max_relations is 0 or negative
            task_logger.info(
                f"Plan ID {spec.plan_id}: Scope is 0 relations as per max_relations."
            )

    # Instantiate the join pattern implementation and generate base joins
    join_pattern_impl = JOIN_PATTERNS[spec.pattern]()
    base_joins = join_pattern_impl.generate_joins(
        rng, plan_scope, plan_config.num_attrs
    )

    # Generate a DOT file for visualization if requested
    if plan_config.gen_dot_viz:
        _generate_dot(
            spec.plan_id,
            spec.pattern,
            plan_scope,
            base_joins,
            spec.output_dir,
        )

    # Warn if the plan scope ended up empty, which will result in minimal output files
    if not plan_scope and spec.max_relations > 0:
        task_logger.warning(
            f"For plan ID {spec.plan_id} (pattern: {spec.pattern}), "
            f"no relations selected for scope (max_relations={spec.max_relations}). "
            f"Derived plan files will be minimal."
        )

    # --- 2. Prepare for analysis if configured ---
    # Check if conditions for running analysis are met
    analysis_possible = (
        plan_config.analyze
        and plan_config.data_output_path
        and "txt" in plan_config.plan_output_format
    )

    # Warn the user if analysis was requested but cannot be performed
    if plan_config.analyze and not analysis_possible:
        task_logger.warning(
            "Analysis requires 'txt' in --plan-output-format and a valid data path. "
            "Skipping analysis for this plan."
        )

    # --- 3. Generate derived plans and (optionally) analyze them ---
    for granularity in plan_config.plan_granularity:
        task_logger.debug(
            f"Plan ID {spec.plan_id}: Deriving for granularity '{granularity}'"
        )

        # Derive the plan stages based on the specified granularity
        plan = _derive_stages(
            plan_id=spec.plan_id,
            pattern=spec.pattern,
            relations=plan_scope,
            joins=base_joins,
            granularity=granularity,
        )

        # Write plan files in all requested output formats
        for format_name in plan_config.plan_output_format:
            plan_formatter_impl = FORMATTERS[format_name]()
            formatted_content = plan_formatter_impl.format(plan)
            plan_file_path = (
                spec.output_dir
                / f"plan_{spec.plan_id}_{spec.pattern}_{granularity}.{format_name}"
            )
            _write_formatted_plan(
                plan_file_path, format_name, formatted_content, task_logger
            )

        # Run analysis for this specific derived plan if possible
        if analysis_possible:
            _run_plan_analysis(
                spec,
                plan_config,
                granularity,
                task_logger,
            )
