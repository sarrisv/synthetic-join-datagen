import json
import logging
import pathlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union, cast, Set

from registries import (
    JOIN_PATTERNS,
    FORMATTERS,
    Plan,
    TableStage,
    AttributeStage,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanConfig:
    """Relevant global CLI arguments for plan generation tasks."""

    plan_granularity: List[str]
    plan_output_format: List[str]
    generate_dot_visualization: bool
    num_attrs: int
    run_on_the_fly_analysis: bool
    data_output_path: Optional[pathlib.Path]
    data_output_format: str
    base_output_path: Optional[pathlib.Path]


@dataclass
class PlanSpec:
    """Specification for generating one fundamental plan structure."""

    plan_id: int
    all_relation_names: List[str]
    pattern: str
    max_relations: int
    output_dir: pathlib.Path
    seed_for_plan: int


def derive_stages(
    fundamental_plan_id: int,
    pattern: str,
    relations: List[str],
    joins: List[Tuple[str, str, int]],
    granularity: str,
) -> Plan:
    """
    Derives staged plan data (TableStage or AttributeStage objects)
    based on granularity from the fundamental binary joins.
    """
    stages: List[Union[TableStage, AttributeStage]] = []

    if granularity == "table":
        if joins:
            aggregated_joins: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for r1, r2, attr_idx in joins:
                pair_key = (r1, r2)
                attr_name = f"attr{attr_idx}"
                if attr_name not in aggregated_joins[pair_key]:
                    aggregated_joins[pair_key].append(attr_name)

            for pair, attrs in sorted(aggregated_joins.items()):
                stages.append(
                    TableStage(
                        table1=pair[0], table2=pair[1], join_attributes=sorted(attrs)
                    )
                )

    elif granularity == "attribute":
        unique_attr_indices = sorted(list(set(attr_idx for _, _, attr_idx in joins)))
        if joins and unique_attr_indices:
            for ua_idx in unique_attr_indices:
                attr_name = f"attr{ua_idx}"
                relations_for_this_stage: Set[str] = set()
                for r1_fj, r2_fj, attr_idx_fj in joins:
                    if attr_idx_fj == ua_idx:
                        relations_for_this_stage.add(r1_fj)
                        relations_for_this_stage.add(r2_fj)

                sorted_rels_for_stage = sorted(list(relations_for_this_stage))
                if sorted_rels_for_stage:
                    stages.append(
                        AttributeStage(
                            join_on_attribute=attr_name,
                            participating_relations=sorted_rels_for_stage,
                        )
                    )

    return Plan(
        plan_id=fundamental_plan_id,
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
    """Generates a DOT file for the given fundamental binary joins."""
    file_path = plan_dir_path / f"plan_{plan_id}_{pattern}_viz.dot"
    task_logger = logging.getLogger(f"{__name__}.dot_viz.{plan_id}.{pattern}")
    task_logger.info(f"Generating DOT visualization: {file_path.name}")

    dot_lines: List[str] = [
        f"graph FundamentalJoins_{plan_id}_{pattern} {{",
        "  graph [",
        '    layout="circo"; overlap=false; center=true;',
        f'    label="Fundamental Joins Visualization\\nPlan ID: {plan_id}\\nPattern: {pattern}\\nScope: {", ".join(relations)}";',
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
            dot_lines.append("\n  // No fundamental binary joins to visualize")
        else:
            edge_labels: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for r1, r2, attr_idx in joins:
                pair_key = (r1, r2)
                attr_name = f"attr{attr_idx}"
                if attr_name not in edge_labels[pair_key]:
                    edge_labels[pair_key].append(attr_name)

            dot_lines.append("")
            for pair, attrs in sorted(edge_labels.items()):
                dot_lines.append(
                    f'  "{pair[0]}" -- "{pair[1]}" [label="{", ".join(sorted(attrs))}"];'
                )

    dot_lines.append("}")
    try:
        with open(file_path, "w") as f:
            f.write("\n".join(dot_lines))
    except IOError as e:
        task_logger.error(f"Could not write DOT visualization file {file_path}: {e}")


def generate_plan(spec: PlanSpec, global_plan_args: PlanConfig) -> None:
    """
    Dask task to generate one fundamental plan and all its derived outputs
    (different granularities and formats).
    """
    task_logger = logging.getLogger(
        f"{__name__}.plan_worker.{spec.plan_id}.{spec.pattern}"
    )
    task_logger.info(
        f"Processing fundamental plan ID {spec.plan_id} with pattern '{spec.pattern}'"
    )

    py_random_instance = random.Random(spec.seed_for_plan)

    num_relations = min(spec.max_relations, len(spec.all_relation_names))
    current_selected_scope: List[str] = []
    if num_relations > 0 and spec.all_relation_names:
        current_selected_scope = sorted(
            py_random_instance.sample(spec.all_relation_names, num_relations)
        )
    elif num_relations == 0:
        task_logger.info(
            f"Plan ID {spec.plan_id}: Scope is 0 relations as per max_relations."
        )
    else:
        task_logger.warning(
            f"Plan ID {spec.plan_id}: No global relations available to select scope from."
        )

    join_pattern_impl = JOIN_PATTERNS[spec.pattern]()
    fundamental_joins_tuples = join_pattern_impl.generate_joins(
        current_selected_scope,
        global_plan_args.num_attrs,
        py_random_instance,
    )

    if global_plan_args.generate_dot_visualization:
        generate_dot(
            spec.plan_id,
            spec.pattern,
            current_selected_scope,
            fundamental_joins_tuples,
            spec.output_dir,
        )

    if not current_selected_scope and spec.max_relations > 0:
        task_logger.warning(
            f"For plan ID {spec.plan_id} (pattern: {spec.pattern}), "
            f"no relations selected for scope (max_relations={spec.max_relations}). "
            f"Derived plan files will be minimal."
        )

    for granularity_to_use in global_plan_args.plan_granularity:
        task_logger.debug(
            f"Plan ID {spec.plan_id}: Deriving for granularity '{granularity_to_use}'"
        )

        plan = derive_stages(
            fundamental_plan_id=spec.plan_id,
            pattern=spec.pattern,
            relations=current_selected_scope,
            joins=fundamental_joins_tuples,
            granularity=granularity_to_use,
        )

        for format_name in global_plan_args.plan_output_format:
            plan_formatter_impl = FORMATTERS[format_name]()
            formatted_body_content = plan_formatter_impl.format(plan)

            file_path_for_plan = (
                spec.output_dir
                / f"plan_{spec.plan_id}_{spec.pattern}_{granularity_to_use}.{format_name}"
            )
            task_logger.debug(
                f"Plan ID {spec.plan_id}: Writing plan file: {file_path_for_plan.name}"
            )

            try:
                with open(file_path_for_plan, "w") as f:
                    common_header_lines = [
                        f"# Execution Plan {plan.plan_id}",
                        f"# Pattern for Fundamental Joins: {plan.pattern}",
                        f"# Output Granularity: {plan.granularity}",
                        f"# Relations in Plan Scope: {', '.join(plan.relations) if plan.relations else 'None'}",
                    ]
                    if not plan.stages:
                        common_header_lines.append(
                            f"# (No specific {plan.granularity} stages derived)"
                        )

                    if format_name == "json":
                        full_json_output = {
                            "plan_id": plan.plan_id,
                            "pattern": plan.pattern,
                            "granularity": plan.granularity,
                            "relations_in_scope": plan.relations,
                            **cast(Dict[str, Any], formatted_body_content),
                        }
                        json.dump(full_json_output, f, indent=2)
                    elif format_name == "txt":
                        f.write("\n".join(common_header_lines) + "\n")
                        f.write(cast(str, formatted_body_content) + "\n")
            except IOError as e:
                task_logger.error(
                    f"Could not write plan file {file_path_for_plan}: {e}"
                )

    # On-the-fly analysis if enabled
    if global_plan_args.run_on_the_fly_analysis and global_plan_args.data_output_path:
        try:
            from analysis_module import (
                analyze_single_plan,
                generate_individual_analysis_report,
            )

            for granularity_to_use in global_plan_args.plan_granularity:
                for format_name in global_plan_args.plan_output_format:
                    if format_name == "txt":  # Only analyze text plans for now
                        plan_file_path = (
                            spec.output_dir
                            / f"plan_{spec.plan_id}_{spec.pattern}_{granularity_to_use}.{format_name}"
                        )

                        if plan_file_path.exists():
                            task_logger.debug(
                                f"Running on-the-fly analysis for {plan_file_path.name}"
                            )

                            analysis = analyze_single_plan(
                                plan_file_path,
                                global_plan_args.data_output_path,
                                global_plan_args.data_output_format,
                            )

                            # Generate individual analysis report
                            individual_report_path = (
                                global_plan_args.base_output_path
                                / f"analysis_plan_{spec.plan_id}_{spec.pattern}_{granularity_to_use}.txt"
                            )

                            generate_individual_analysis_report(
                                analysis, individual_report_path
                            )
                            task_logger.info(
                                f"On-the-fly analysis written to: {individual_report_path.name}"
                            )

        except Exception as e:
            task_logger.warning(
                f"On-the-fly analysis failed for plan {spec.plan_id}: {e}"
            )
