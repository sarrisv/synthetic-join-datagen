# plan_generation_module.py
"""
Handles the generation of fundamental plan structures and the derivation
of staged plan outputs (table or attribute granularity).
Also includes DOT visualization for fundamental plans.
"""

import pathlib
import random
import logging
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, Union, Type, cast, Set
from dataclasses import dataclass

from registries import (
    JOIN_PATTERNS,
    PLAN_FORMATTERS,
    JoinPattern,
    PlanFormatter,
    DerivedPlanOutput,
    TablePlanStage,
    AttributePlanStage,
)

logger = logging.getLogger(__name__)
INDENT: str = "  "


@dataclass
class GlobalCLIArgsPlanSubset:
    """Relevant global CLI arguments for plan generation tasks."""

    plan_granularity: List[str]
    plan_output_format: List[str]
    generate_dot_visualization: bool
    num_attributes_per_relation: int
    run_on_the_fly_analysis: bool
    data_output_path: Optional[pathlib.Path]
    data_output_format: str
    base_output_path: Optional[pathlib.Path]


@dataclass
class FundamentalPlanSpecForWorker:
    """Specification for generating one fundamental plan structure."""

    plan_id: int
    all_relation_names: List[str]
    join_pattern_name: str
    max_relations_in_scope: int
    plan_output_dir: pathlib.Path
    seed_for_plan: int


def derive_staged_plan_data_from_fundamental(
    fundamental_plan_id: int,
    join_pattern_name: str,
    selected_relations_in_scope: List[str],
    fundamental_binary_joins: List[Tuple[str, str, int]],
    granularity: str,
) -> DerivedPlanOutput:
    """
    Derives staged plan data (TablePlanStage or AttributePlanStage objects)
    based on granularity from the fundamental binary joins.
    """
    stages: List[Union[TablePlanStage, AttributePlanStage]] = []

    if granularity == "table":
        if fundamental_binary_joins:
            aggregated_joins: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for r1, r2, attr_idx in fundamental_binary_joins:
                pair_key = (r1, r2)
                attr_name = f"attr{attr_idx}"
                if attr_name not in aggregated_joins[pair_key]:
                    aggregated_joins[pair_key].append(attr_name)

            for pair, attrs in sorted(aggregated_joins.items()):
                stages.append(
                    TablePlanStage(
                        table1=pair[0], table2=pair[1], join_attributes=sorted(attrs)
                    )
                )

    elif granularity == "attribute":
        unique_attr_indices = sorted(
            list(set(attr_idx for _, _, attr_idx in fundamental_binary_joins))
        )
        if fundamental_binary_joins and unique_attr_indices:
            for ua_idx in unique_attr_indices:
                attr_name = f"attr{ua_idx}"
                relations_for_this_stage: Set[str] = set()
                for r1_fj, r2_fj, attr_idx_fj in fundamental_binary_joins:
                    if attr_idx_fj == ua_idx:
                        relations_for_this_stage.add(r1_fj)
                        relations_for_this_stage.add(r2_fj)

                sorted_rels_for_stage = sorted(list(relations_for_this_stage))
                if sorted_rels_for_stage:
                    stages.append(
                        AttributePlanStage(
                            join_on_attribute=attr_name,
                            participating_relations=sorted_rels_for_stage,
                        )
                    )

    return DerivedPlanOutput(
        plan_id=fundamental_plan_id,
        join_pattern_name=join_pattern_name,
        granularity=granularity,
        selected_relations_in_scope=selected_relations_in_scope,
        stages=stages,
    )


def generate_dot_visualization_file_for_fundamental(
    plan_id: int,
    join_pattern_name: str,
    selected_relations_in_scope: List[str],
    fundamental_binary_joins: List[Tuple[str, str, int]],
    plan_dir_path: pathlib.Path,
) -> None:
    """Generates a DOT file for the given fundamental binary joins."""
    file_path = plan_dir_path / f"plan_{plan_id}_{join_pattern_name}_viz.dot"
    task_logger = logging.getLogger(
        f"{__name__}.dot_viz.{plan_id}.{join_pattern_name}"
    )
    task_logger.info(f"Generating DOT visualization: {file_path.name}")

    dot_lines: List[str] = [
        f"graph FundamentalJoins_{plan_id}_{join_pattern_name} {{",
        f"{INDENT}graph [",
        f'{INDENT * 2}layout="circo"; overlap=false; center=true;',
        f'{INDENT * 2}label="Fundamental Joins Visualization\\nPlan ID: {plan_id}\\nPattern: {join_pattern_name}\\nScope: {", ".join(selected_relations_in_scope)}";',
        f"{INDENT * 2}labelloc=t; fontsize=10;",
        f"{INDENT}]",
        f"\n{INDENT}node [shape=ellipse, fontsize=10];",
        f"{INDENT}edge [fontsize=8];",
    ]

    if not selected_relations_in_scope:
        dot_lines.append(f"\n{INDENT}// No relations in scope for visualization")
    else:
        dot_lines.append("")
        for rel_name in selected_relations_in_scope:
            dot_lines.append(f'{INDENT}"{rel_name}";')

        if not fundamental_binary_joins:
            dot_lines.append(f"\n{INDENT}// No fundamental binary joins to visualize")
        else:
            edge_labels: Dict[Tuple[str, str], List[str]] = defaultdict(list)
            for r1, r2, attr_idx in fundamental_binary_joins:
                pair_key = (r1, r2)
                attr_name = f"attr{attr_idx}"
                if attr_name not in edge_labels[pair_key]:
                    edge_labels[pair_key].append(attr_name)

            dot_lines.append("")
            for pair, attrs in sorted(edge_labels.items()):
                dot_lines.append(
                    f'{INDENT}"{pair[0]}" -- "{pair[1]}" [label="{", ".join(sorted(attrs))}"];'
                )

    dot_lines.append("}")
    try:
        with open(file_path, "w") as f:
            f.write("\n".join(dot_lines))
    except IOError as e:
        task_logger.error(f"Could not write DOT visualization file {file_path}: {e}")


def process_one_fundamental_plan_task(
    spec: FundamentalPlanSpecForWorker, global_plan_args: GlobalCLIArgsPlanSubset
) -> None:
    """
    Dask task to generate one fundamental plan and all its derived outputs
    (different granularities and formats).
    """
    task_logger = logging.getLogger(
        f"{__name__}.plan_worker.{spec.plan_id}.{spec.join_pattern_name}"
    )
    task_logger.info(
        f"Processing fundamental plan ID {spec.plan_id} with pattern \'{spec.join_pattern_name}\'"
    )

    py_random_instance = random.Random(spec.seed_for_plan)

    num_rels_for_this_scope = min(
        spec.max_relations_in_scope, len(spec.all_relation_names)
    )
    current_selected_scope: List[str] = []
    if num_rels_for_this_scope > 0 and spec.all_relation_names:
        current_selected_scope = sorted(
            py_random_instance.sample(spec.all_relation_names, num_rels_for_this_scope)
        )
    elif num_rels_for_this_scope == 0:
        task_logger.info(
            f"Plan ID {spec.plan_id}: Scope is 0 relations as per max_relations_in_scope."
        )
    else:
        task_logger.warning(
            f"Plan ID {spec.plan_id}: No global relations available to select scope from."
        )

    join_pattern_impl = JOIN_PATTERNS[spec.join_pattern_name]()
    fundamental_joins_tuples = join_pattern_impl.generate_fundamental_joins(
        current_selected_scope,
        global_plan_args.num_attributes_per_relation,
        py_random_instance,
    )

    if global_plan_args.generate_dot_visualization:
        generate_dot_visualization_file_for_fundamental(
            spec.plan_id,
            spec.join_pattern_name,
            current_selected_scope,
            fundamental_joins_tuples,
            spec.plan_output_dir,
        )

    if not current_selected_scope and spec.max_relations_in_scope > 0:
        task_logger.warning(
            f"For plan ID {spec.plan_id} (pattern: {spec.join_pattern_name}), "
            f"no relations selected for scope (max_relations_in_scope={spec.max_relations_in_scope}). "
            f"Derived plan files will be minimal."
        )

    for granularity_to_use in global_plan_args.plan_granularity:
        task_logger.debug(
            f"Plan ID {spec.plan_id}: Deriving for granularity '{granularity_to_use}'"
        )

        derived_plan_output_obj = derive_staged_plan_data_from_fundamental(
            fundamental_plan_id=spec.plan_id,
            join_pattern_name=spec.join_pattern_name,
            selected_relations_in_scope=current_selected_scope,
            fundamental_binary_joins=fundamental_joins_tuples,
            granularity=granularity_to_use,
        )

        for format_name in global_plan_args.plan_output_format:
            plan_formatter_impl = PLAN_FORMATTERS[format_name]()
            formatted_body_content = plan_formatter_impl.format_plan_body(
                derived_plan_output_obj
            )

            file_path_for_plan = (
                spec.plan_output_dir
                / f"plan_{spec.plan_id}_{spec.join_pattern_name}_{granularity_to_use}.{format_name}"
            )
            task_logger.debug(
                f"Plan ID {spec.plan_id}: Writing plan file: {file_path_for_plan.name}"
            )

            try:
                with open(file_path_for_plan, "w") as f:
                    common_header_lines = [
                        f"# Execution Plan {derived_plan_output_obj.plan_id}",
                        f"# Pattern for Fundamental Joins: {derived_plan_output_obj.join_pattern_name}",
                        f"# Output Granularity: {derived_plan_output_obj.granularity}",
                        f"# Relations in Plan Scope: {', '.join(derived_plan_output_obj.selected_relations_in_scope) if derived_plan_output_obj.selected_relations_in_scope else 'None'}",
                    ]
                    if not derived_plan_output_obj.stages:
                        common_header_lines.append(
                            f"# (No specific {derived_plan_output_obj.granularity} stages derived)"
                        )

                    if format_name == "json":
                        full_json_output = {
                            "plan_id": derived_plan_output_obj.plan_id,
                            "pattern": derived_plan_output_obj.join_pattern_name,
                            "granularity": derived_plan_output_obj.granularity,
                            "relations_in_scope": derived_plan_output_obj.selected_relations_in_scope,
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
            from analysis_module import analyze_single_plan, generate_individual_analysis_report

            for granularity_to_use in global_plan_args.plan_granularity:
                for format_name in global_plan_args.plan_output_format:
                    if format_name == "txt":  # Only analyze text plans for now
                        plan_file_path = (
                            spec.plan_output_dir
                            / f"plan_{spec.plan_id}_{spec.join_pattern_name}_{granularity_to_use}.{format_name}"
                        )

                        if plan_file_path.exists():
                            task_logger.debug(f"Running on-the-fly analysis for {plan_file_path.name}")

                            analysis = analyze_single_plan(
                                plan_file_path,
                                global_plan_args.data_output_path,
                                global_plan_args.data_output_format
                            )

                            # Generate individual analysis report
                            individual_report_path = (
                                global_plan_args.base_output_path
                                / f"analysis_plan_{spec.plan_id}_{spec.join_pattern_name}_{granularity_to_use}.txt"
                            )

                            generate_individual_analysis_report(analysis, individual_report_path)
                            task_logger.info(f"On-the-fly analysis written to: {individual_report_path.name}")

        except Exception as e:
            task_logger.warning(f"On-the-fly analysis failed for plan {spec.plan_id}: {e}")
