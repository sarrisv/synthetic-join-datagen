import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union, cast, Set

import numpy as np

from registries import JOIN_PATTERNS, FORMATTERS
from plan_structures import Plan, TableStage, AttributeStage

logger = logging.getLogger(__name__)


@dataclass
class PlanConfig:
    """Relevant global CLI arguments for plan generation tasks"""

    plan_granularity: List[str]
    plan_output_format: List[str]
    generate_dot_visualization: bool
    num_attrs: int
    run_on_the_fly_analysis: bool
    data_output_path: Optional[pathlib.Path]
    data_output_format: str
    base_output_path: Optional[pathlib.Path]
    analysis_subdir: str


@dataclass
class PlanSpec:
    """Specification for a base plan"""

    plan_id: int
    relations: List[str]
    pattern: str
    max_relations: int
    output_dir: pathlib.Path
    seed: int


def derive_stages(
    plan_id: int,
    relations: List[str],
    pattern: str,
    joins: List[Tuple[str, str, int]],
    granularity: str,
) -> Plan:
    """
    Derives staged plan data (TableStage or AttributeStage objects)
    based on granularity from the base join
    """
    stages: List[Union[TableStage, AttributeStage]] = []

    if not joins:
        # If there are no base joins, there are no stages to derive.
        return Plan(
            plan_id=plan_id,
            pattern=pattern,
            granularity=granularity,
            relations=relations,
            stages=[],
        )

    if granularity == "table":
        # Group unique attributes by the relation pair they join.
        aggregated_joins: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for r1, r2, attr_idx in joins:
            aggregated_joins[(r1, r2)].add(f"attr{attr_idx}")

        # Create a sorted list of TableStage objects from the aggregated data.
        for pair, attrs_set in sorted(aggregated_joins.items()):
            stages.append(
                TableStage(
                    rel1=pair[0], rel2=pair[1], join_attributes=sorted(list(attrs_set))
                )
            )

    elif granularity == "attribute":
        # Group participating relations by the attribute they are joined on.
        relations_by_attr: Dict[str, Set[str]] = defaultdict(set)
        for r1, r2, attr_idx in joins:
            attr_name = f"attr{attr_idx}"
            relations_by_attr[attr_name].add(r1)
            relations_by_attr[attr_name].add(r2)

        # Create a sorted list of AttributeStage objects.
        for attr_name, relations_set in sorted(relations_by_attr.items()):
            stages.append(
                AttributeStage(
                    joining_attribute=attr_name,
                    participating_relations=sorted(list(relations_set)),
                )
            )

    return Plan(
        plan_id=plan_id,
        pattern=pattern,
        granularity=granularity,
        relations=relations,
        stages=stages,
    )


def generate_dot(
    plan_id: int,
    pattern: str,
    relations: List[str],
    joins: List[Tuple[str, str, int]],
    plan_dir_path: pathlib.Path,
) -> None:
    """Generates a DOT file for the given base binary joins."""
    file_path = plan_dir_path / f"plan_{plan_id}_{pattern}_base_joins_viz.dot"
    task_logger = logging.getLogger(f"{__name__}.dot_viz.{plan_id}.{pattern}")
    task_logger.info(f"Generating DOT visualization: {file_path.name}")

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
        dot_lines.append("")
        for rel_name in relations:
            dot_lines.append(f'  "{rel_name}";')

        if not joins:
            dot_lines.append("\n  // No base binary joins to visualize")
        else:
            edge_labels: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            for r1, r2, attr_idx in joins:
                attr_name = f"attr{attr_idx}"
                edge_labels[(r1, r2)].add(attr_name)

            dot_lines.append("")
            for pair, attrs_set in sorted(edge_labels.items()):
                attrs_list = sorted(list(attrs_set))
                dot_lines.append(
                    f'  "{pair[0]}" -- "{pair[1]}" [label="{", ".join(attrs_list)}"];'
                )

    dot_lines.append("}")
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
    """Writes formatted plan content to a file, handling different formats."""
    task_logger.debug(f"Writing plan to {file_path.name}")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if format_name == "json":
                # The content is expected to be a dict here.
                json.dump(formatted_content, f, indent=2)
            elif format_name == "txt":
                # The content is expected to be a string here.
                f.write(cast(str, formatted_content) + "\n")
    except IOError as e:
        task_logger.error(f"Could not write plan file {file_path}: {e}")


def _run_on_the_fly_analysis(
    spec: PlanSpec,
    plan_config: PlanConfig,
    granularity: str,
    analysis_helpers: Tuple[Any, Any],
    task_logger: logging.Logger,
) -> None:
    """Runs and saves the on-the-fly analysis for a specific generated plan."""
    analyze_single_plan, save_plan_analysis_to_json = analysis_helpers

    plan_txt_path = (
        spec.output_dir / f"plan_{spec.plan_id}_{spec.pattern}_{granularity}.txt"
    )

    if not plan_txt_path.exists():
        task_logger.warning(
            f"Plan file {plan_txt_path.name} not found, skipping its analysis."
        )
        return

    task_logger.debug(f"Running on-the-fly analysis for {plan_txt_path.name}")
    try:
        analysis = analyze_single_plan(
            plan_txt_path,
            cast(pathlib.Path, plan_config.data_output_path),
            plan_config.data_output_format,
        )

        analysis_output_dir = (
            cast(pathlib.Path, plan_config.base_output_path)
            / plan_config.analysis_subdir
        )
        individual_json_path = (
            analysis_output_dir
            / f"analysis_plan_{spec.plan_id}_{spec.pattern}_{granularity}.json"
        )

        save_plan_analysis_to_json(analysis, individual_json_path)
        task_logger.info(
            f"On-the-fly analysis JSON written to: {individual_json_path.name}"
        )
    except Exception as e:
        task_logger.warning(
            f"On-the-fly analysis failed for plan {spec.plan_id}, granularity '{granularity}': {e}"
        )


def generate_plan(spec: PlanSpec, plan_config: PlanConfig) -> None:
    """
    Dask task to generate one base plan and all its derived outputs
    (different granularities and formats).
    """
    task_logger = logging.getLogger(
        f"{__name__}.plan_worker.{spec.plan_id}.{spec.pattern}"
    )
    task_logger.info(
        f"Processing base plan ID {spec.plan_id} with pattern '{spec.pattern}'"
    )
    rng = np.random.default_rng(spec.seed)

    # --- 1. Select relations for the plan scope ---
    current_selected_scope: List[str] = []
    if not spec.relations:
        if spec.max_relations > 0:
            task_logger.warning(
                f"Plan ID {spec.plan_id}: No global relations available to select scope from."
            )
    else:
        num_relations_to_select = min(spec.max_relations, len(spec.relations))
        if num_relations_to_select > 0:
            current_selected_scope = sorted(
                rng.choice(
                    spec.relations, size=num_relations_to_select, replace=False
                ).tolist()
            )
        else:  # This happens if spec.max_relations is 0 or negative
            task_logger.info(
                f"Plan ID {spec.plan_id}: Scope is 0 relations as per max_relations."
            )

    join_pattern_impl = JOIN_PATTERNS[spec.pattern]()
    base_joins = join_pattern_impl.generate_joins(
        rng, current_selected_scope, plan_config.num_attrs
    )

    if plan_config.generate_dot_visualization:
        generate_dot(
            spec.plan_id,
            spec.pattern,
            current_selected_scope,
            base_joins,
            spec.output_dir,
        )

    if not current_selected_scope and spec.max_relations > 0:
        task_logger.warning(
            f"For plan ID {spec.plan_id} (pattern: {spec.pattern}), "
            f"no relations selected for scope (max_relations={spec.max_relations}). "
            f"Derived plan files will be minimal."
        )

    # --- 2. Prepare for on-the-fly analysis if configured ---
    analysis_possible = (
        plan_config.run_on_the_fly_analysis
        and plan_config.data_output_path
        and "txt" in plan_config.plan_output_format
    )
    analysis_helpers = None

    if plan_config.run_on_the_fly_analysis and not analysis_possible:
        task_logger.warning(
            "On-the-fly analysis requires 'txt' in --plan-output-format and a valid data path. "
            "Skipping analysis for this plan."
        )
    elif analysis_possible:
        try:
            # Import lazily to avoid dependency if not used.
            from analysis_module import analyze_single_plan, save_plan_analysis_to_json

            analysis_helpers = (analyze_single_plan, save_plan_analysis_to_json)
        except ImportError as e:
            task_logger.warning(
                f"Could not import analysis module, skipping analysis: {e}"
            )
            analysis_possible = False

    # --- 3. Generate derived plans and (optionally) analyze them ---
    for granularity in plan_config.plan_granularity:
        task_logger.debug(
            f"Plan ID {spec.plan_id}: Deriving for granularity '{granularity}'"
        )

        plan = derive_stages(
            plan_id=spec.plan_id,
            pattern=spec.pattern,
            relations=current_selected_scope,
            joins=base_joins,
            granularity=granularity,
        )

        # Write plan files in all requested formats
        for format_name in plan_config.plan_output_format:
            plan_formatter_impl = FORMATTERS[format_name]()
            formatted_content = plan_formatter_impl.format(plan)
            file_path_for_plan = (
                spec.output_dir
                / f"plan_{spec.plan_id}_{spec.pattern}_{granularity}.{format_name}"
            )
            _write_formatted_plan(
                file_path_for_plan, format_name, formatted_content, task_logger
            )

        # Run on-the-fly analysis for this granularity if configured
        if analysis_possible and analysis_helpers:
            _run_on_the_fly_analysis(
                spec,
                plan_config,
                granularity,
                analysis_helpers,
                task_logger,
            )
