import json
import logging
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np

from plan_structures import BinaryJoinStage, MultiwayJoinStage

logger = logging.getLogger("analysis")


@dataclass
class AttributeMetadata:
    """Metadata for a single attribute in a relation"""

    null_count: int
    distinct_values: int
    value_counts: Dict[str, int]


@dataclass
class RelationMetadata:
    """Metadata for a full relation, including all its attributes"""

    name: str
    total_rows: int
    attributes: Dict[str, AttributeMetadata]


@dataclass
class JoinStageStats:
    """Statistics and metrics for a single join operation (stage)"""

    join_attribute: str
    relations: List[str]
    relation_sizes: Dict[str, int]
    distinct_values_per_relation: Dict[str, int]
    common_values_across_all: int
    selectivities: Dict[str, float]
    null_counts_per_relation: Dict[str, int]
    result_size: int
    reduction_factor: float
    common_values_list: List[str]


@dataclass
class JoinPlanStats:
    """Aggregated statistics for a complete execution plan"""

    plan_id: int
    pattern: str
    granularity: str
    relations: List[str]
    stage_stats: List[JoinStageStats]
    total_intermediate_size: int
    average_selectivity: float
    max_selectivity: float
    min_selectivity: float
    execution_model: str  # "table-at-a-time" or "attribute-at-a-time"


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively converts numpy types in an object to native Python types for JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    return obj


def save_plan_analysis(analysis: JoinPlanStats, output_path: pathlib.Path) -> None:
    """Saves a single plan's analysis results to a JSON file"""
    analysis_dict = asdict(analysis)
    analysis_dict_native = _convert_numpy_types(analysis_dict)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict_native, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved individual analysis to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save individual analysis JSON to {output_path}: {e}")


def _load_relation_metadata(
    data_path: pathlib.Path, relation_name: str
) -> RelationMetadata:
    """Loads and parses the metadata file for a single relation"""
    metadata_path = (data_path / relation_name).with_suffix(".meta.json")
    logger.debug(f"Loading relation metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    attributes = {
        attr_name: AttributeMetadata(**attr_data)
        for attr_name, attr_data in data["attributes"].items()
    }
    data["attributes"] = attributes
    return RelationMetadata(**data)


def _parse_plan_file(
    plan_path: pathlib.Path,
) -> Tuple[Dict[str, Any], Union[List[BinaryJoinStage], List[MultiwayJoinStage]]]:
    """Parses a plan file, extracting metadata and join stage specifications"""
    metadata: Dict[str, Any] = {}
    table_joins: List[BinaryJoinStage] = []
    attr_stages: List[MultiwayJoinStage] = []
    parsing_stages = False

    with open(plan_path, "r", encoding="utf-8") as f:
        # State machine to switch from parsing metadata to parsing stages
        for line in f:
            line = line.strip()
            if not line:
                continue

            if not parsing_stages:
                # --- Parse metadata header (lines starting with '#') ---
                if line.startswith("# Execution Plan"):
                    metadata["plan_id"] = int(line.split()[-1])
                elif line.startswith("# Pattern for Base Joins:"):
                    metadata["pattern"] = line.split(": ", 1)[1]
                elif line.startswith("# Output Granularity:"):
                    metadata["granularity"] = line.split(": ", 1)[1]
                elif line.startswith("# Relations in Plan Scope:"):
                    scope_part = line.split(": ", 1)[1].strip()
                    if scope_part == "None":
                        metadata["relations"] = []
                    else:
                        metadata["relations"] = [
                            r.strip() for r in scope_part.split(",") if r.strip()
                        ]
                # Mark the end of the header and start of stages
                elif line.startswith(
                    ("# Table-at-a-Time Stages:", "# Attribute-at-a-Time Stages:")
                ):
                    parsing_stages = True
            elif line and not line.startswith("#"):
                # --- Parse stage definition lines ---
                parts = line.split(",")
                if metadata.get("granularity") == "table" and len(parts) >= 3:
                    # Format: R0,R2,attr3[,attr4,...]
                    rel1, rel2 = parts[0], parts[1]
                    join_attributes = [attr.strip() for attr in parts[2:]]
                    table_joins.append(BinaryJoinStage(rel1, rel2, join_attributes))
                elif metadata.get("granularity") == "attribute" and len(parts) >= 2:
                    # Format: attr1,R0[,R1,...]
                    join_attr = parts[0]
                    relations = [r.strip() for r in parts[1:]]
                    attr_stages.append(
                        MultiwayJoinStage(
                            joining_attribute=join_attr, relations=relations
                        )
                    )

    if metadata.get("granularity") == "attribute":
        return metadata, attr_stages
    return metadata, table_joins


def _create_intermediate_metadata(
    left_meta: RelationMetadata,
    right_meta: RelationMetadata,
    join_attr: str,
    stage_stats: JoinStageStats,
) -> RelationMetadata:
    """Creates metadata for an intermediate join result, scaling non-join attributes"""
    new_name = f"({left_meta.name}⋈{right_meta.name})"
    new_total_rows = stage_stats.result_size
    new_attributes: Dict[str, AttributeMetadata] = {}

    # Calculate new metadata for the joining attribute (exact calculation)
    left_join_attr_meta = left_meta.attributes[join_attr]
    right_join_attr_meta = right_meta.attributes[join_attr]

    # Calculate exact value distribution for the join attribute: new_count = count_left * count_right
    new_join_attr_value_counts = {
        v: left_join_attr_meta.value_counts.get(str(v), 0)
        * right_join_attr_meta.value_counts.get(str(v), 0)
        for v in stage_stats.common_values_list
    }
    # Filter out values that no longer appear after the join
    new_join_attr_value_counts = {
        k: v for k, v in new_join_attr_value_counts.items() if v > 0
    }

    new_attributes[join_attr] = AttributeMetadata(
        null_count=0,  # Nulls don't participate in an inner join
        distinct_values=len(new_join_attr_value_counts),
        value_counts=new_join_attr_value_counts,
    )

    # Estimate metadata for non-join attributes, assuming uniform distribution
    for source_meta in [left_meta, right_meta]:
        if source_meta.total_rows > 0:
            # Calculate the reduction factor for scaling this relation's attributes
            reduction = new_total_rows / source_meta.total_rows
            for attr_name, attr_meta in source_meta.attributes.items():
                # Skip the join attribute, as it was handled separately
                if attr_name == join_attr:
                    continue

                # Scale distinct value counts by the reduction factor
                scaled_counts = {
                    k: int(round(v * reduction))
                    for k, v in attr_meta.value_counts.items()
                }
                # Remove values with counts that rounded to zero
                scaled_counts = {k: v for k, v in scaled_counts.items() if v > 0}

                # Also scale the null count by the same factor
                new_attributes[attr_name] = AttributeMetadata(
                    null_count=int(round(attr_meta.null_count * reduction)),
                    distinct_values=len(scaled_counts),
                    value_counts=scaled_counts,
                )

    return RelationMetadata(
        name=new_name, total_rows=new_total_rows, attributes=new_attributes
    )


def _compute_attribute_stage_selectivity(
    relation_metadata: Dict[str, RelationMetadata],
    join_attribute: str,
    participating_relations: List[str],
) -> JoinStageStats:
    """Calculates cardinality and stats for a multi-way join on a single attribute"""
    relation_value_keys = {}
    relation_sizes = {}
    distinct_values_per_relation = {}
    null_counts_per_relation = {}

    # Collect metadata for all relations involved in the multi-way join
    for rel_name in participating_relations:
        if rel_name in relation_metadata:
            rel_meta = relation_metadata[rel_name]
            relation_sizes[rel_name] = rel_meta.total_rows
            null_counts_per_relation[rel_name] = 0  # Default

            if join_attribute in rel_meta.attributes:
                attr_meta = rel_meta.attributes[join_attribute]

                # The keys of the value_counts dict are the distinct non-null values
                relation_value_keys[rel_name] = attr_meta.value_counts.keys()
                distinct_values_per_relation[rel_name] = attr_meta.distinct_values
                null_counts_per_relation[rel_name] = attr_meta.null_count
            else:
                relation_value_keys[rel_name] = set()
                distinct_values_per_relation[rel_name] = 0

    # Find values common to all participating relations
    if relation_value_keys:
        # Use set intersection to find join keys present in every relation
        sets_to_intersect = (set(keys) for keys in relation_value_keys.values())
        common_values_set = set.intersection(*sets_to_intersect)
    else:
        common_values_set = set()

    common_values_count = len(common_values_set)

    # Calculate per-relation selectivity: the fraction of tuples that will join
    selectivities = {}
    if common_values_set:
        for rel_name in participating_relations:
            if rel_name in relation_metadata and relation_sizes[rel_name] > 0:
                rel_meta = relation_metadata[rel_name]
                if join_attribute in rel_meta.attributes:
                    # Sum counts of common values to find total participating tuples
                    value_counts = rel_meta.attributes[join_attribute].value_counts
                    participating_tuples = sum(
                        value_counts.get(str(v), 0) for v in common_values_set
                    )
                    selectivities[rel_name] = (
                        participating_tuples / relation_sizes[rel_name]
                    )
                else:
                    selectivities[rel_name] = 0.0
            else:
                selectivities[rel_name] = 0.0

    # Calculate exact result size: Σ_v [ Count(v, R1) * ... * Count(v, Rn) ]
    value_counts_per_relation = []

    # Check if all relations have the required join attribute
    all_relations_valid = True
    for rel_name in participating_relations:
        if not (
            rel_name in relation_metadata
            and join_attribute in relation_metadata[rel_name].attributes
        ):
            all_relations_valid = False
            break
        value_counts_per_relation.append(
            relation_metadata[rel_name].attributes[join_attribute].value_counts
        )

    result_size = 0
    if all_relations_valid and common_values_set:
        # For each common value, multiply its frequency across relations and sum the products
        for v in common_values_set:
            product = 1
            for rel_counts in value_counts_per_relation:
                product *= rel_counts.get(str(v), 0)
            result_size += product

    # Reduction factor is the result size relative to the Cartesian product size
    cartesian_size = 1
    for rel_name in participating_relations:
        cartesian_size *= relation_sizes.get(rel_name, 1)

    reduction_factor = result_size / cartesian_size if cartesian_size > 0 else 0.0

    return JoinStageStats(
        join_attribute=join_attribute,
        relations=participating_relations,
        relation_sizes=relation_sizes,
        distinct_values_per_relation=distinct_values_per_relation,
        common_values_across_all=common_values_count,
        selectivities=selectivities,
        null_counts_per_relation=null_counts_per_relation,
        result_size=result_size,
        reduction_factor=reduction_factor,
        common_values_list=sorted(list(common_values_set)),
    )


def _compute_join_selectivity(
    rel1_meta: RelationMetadata,
    rel2_meta: RelationMetadata,
    join_attribute: str,
) -> JoinStageStats:
    """Calculates cardinality and stats for a binary join between two relations"""

    attr = join_attribute

    rel1_size = rel1_meta.total_rows
    rel2_size = rel2_meta.total_rows

    rel1_attr_meta = rel1_meta.attributes.get(attr)
    rel2_attr_meta = rel2_meta.attributes.get(attr)

    if not rel1_attr_meta or not rel2_attr_meta:
        raise ValueError(f"Join attribute '{attr}' metadata not found for join.")

    # Distinct values
    rel1_distinct_keys = rel1_attr_meta.value_counts.keys()
    rel2_distinct_keys = rel2_attr_meta.value_counts.keys()
    common_values_set = set(rel1_distinct_keys).intersection(rel2_distinct_keys)

    common_values_count = len(common_values_set)

    # Calculate selectivity for rel1: sum counts of common values to find matching tuples
    rel1_participating = sum(
        rel1_attr_meta.value_counts.get(v, 0) for v in common_values_set
    )
    rel1_selectivity = rel1_participating / rel1_size if rel1_size > 0 else 0.0

    # Do the same for rel2
    rel2_participating = sum(
        rel2_attr_meta.value_counts.get(v, 0) for v in common_values_set
    )
    rel2_selectivity = rel2_participating / rel2_size if rel2_size > 0 else 0.0

    # Calculate exact join size: Σ_v [ Count(v, R1) * Count(v, R2) ]
    result_size = sum(
        rel1_attr_meta.value_counts.get(v, 0) * rel2_attr_meta.value_counts.get(v, 0)
        for v in common_values_set
    )

    cartesian_size = rel1_size * rel2_size
    reduction_factor = result_size / cartesian_size if cartesian_size > 0 else 0.0

    return JoinStageStats(
        join_attribute=join_attribute,
        relations=[rel1_meta.name, rel2_meta.name],
        relation_sizes={rel1_meta.name: rel1_size, rel2_meta.name: rel2_size},
        distinct_values_per_relation={
            rel1_meta.name: rel1_attr_meta.distinct_values,
            rel2_meta.name: rel2_attr_meta.distinct_values,
        },
        common_values_across_all=common_values_count,
        selectivities={
            rel1_meta.name: rel1_selectivity,
            rel2_meta.name: rel2_selectivity,
        },
        result_size=result_size,
        reduction_factor=reduction_factor,
        null_counts_per_relation={
            rel1_meta.name: rel1_attr_meta.null_count,
            rel2_meta.name: rel2_attr_meta.null_count,
        },
        common_values_list=sorted(list(common_values_set)),
    )


def _calculate_aggregate_metrics(
    selectivities: List[float],
) -> Tuple[float, float, float]:
    """Computes mean, max, and min for a list of selectivities"""
    if not selectivities:
        return 0.0, 0.0, 0.0
    return (
        float(np.mean(selectivities)),
        float(np.max(selectivities)),
        float(np.min(selectivities)),
    )


def analyze_single_plan(
    plan_path: pathlib.Path, data_path: pathlib.Path
) -> JoinPlanStats:
    """Analyzes a single execution plan file by calculating cardinalities and selectivities"""

    plan_logger = logging.getLogger(f"{__name__}.analyze_plan.{plan_path.stem}")
    plan_logger.debug(f"Analyzing plan: {plan_path.name}")

    metadata, stages_spec = _parse_plan_file(plan_path)
    granularity = metadata.get("granularity", "unknown")

    # Load metadata for all relations in the plan's scope
    relations_in_scope = metadata.get("relations", [])
    relation_metadata = {}

    for relation in relations_in_scope:
        try:
            # Load each relation's metadata from its corresponding JSON file
            relation_metadata[relation] = _load_relation_metadata(data_path, relation)
        except Exception as e:
            plan_logger.warning(f"Could not load metadata for relation {relation}: {e}")
            continue

    stage_analysis: List[JoinStageStats] = []
    execution_model = "unknown"

    if granularity == "table":
        # Handle "table-at-a-time" execution as a sequence of pipelined binary joins
        execution_model = "table-at-a-time"
        plan_logger.info(
            "Using sequential analysis model for table-at-a-time plan (assumes left-deep join tree)."
        )
        join_stages = cast(List[BinaryJoinStage], stages_spec)

        # `live_metadata` tracks metadata for base tables and intermediate results
        live_metadata = relation_metadata.copy()
        # `intermediate_meta` holds the result of the previous join stage
        intermediate_meta: Optional[RelationMetadata] = None
        # `joined_relations` tracks which base relations have been joined
        joined_relations: Set[str] = set()

        for stage in join_stages:
            # Determine the inputs for this stage's join
            left_meta: Optional[RelationMetadata] = None
            right_meta: Optional[RelationMetadata] = None

            if not intermediate_meta:
                # The first join uses two base relations
                left_meta = live_metadata.get(stage.rel1)
                right_meta = live_metadata.get(stage.rel2)
                if left_meta and right_meta:
                    joined_relations.update([left_meta.name, right_meta.name])
            else:
                # Subsequent joins use the prior result and a new, un-joined base relation
                left_meta = intermediate_meta

                # Find the relation from the current stage that is not yet in the joined set
                new_relation_name = None
                if stage.rel1 not in joined_relations:
                    new_relation_name = stage.rel1
                elif stage.rel2 not in joined_relations:
                    new_relation_name = stage.rel2

                if new_relation_name:
                    right_meta = live_metadata.get(new_relation_name)
                    if right_meta:
                        joined_relations.add(right_meta.name)
                else:
                    # This case handles invalid plans for a left-deep tree (eg, cycles)
                    plan_logger.warning(
                        f"Could not find a new relation for stage ({stage.rel1}, {stage.rel2}). "
                        f"Both relations may already be in the join pipeline: {joined_relations}. Skipping stage."
                    )
                    continue

            if not left_meta or not right_meta:
                plan_logger.warning("Missing metadata for join inputs. Skipping stage.")
                continue

            # For composite keys, analyze each attribute but only propagate metadata from the last one (a simplification)
            for attr in stage.join_attributes:
                try:
                    stats = _compute_join_selectivity(left_meta, right_meta, attr)
                    stage_analysis.append(stats)

                    # Create output metadata to be used as input for the next stage
                    intermediate_meta = _create_intermediate_metadata(
                        left_meta, right_meta, attr, stats
                    )
                    live_metadata[intermediate_meta.name] = intermediate_meta

                except Exception as e:
                    plan_logger.warning(
                        f"Could not compute selectivity for join {left_meta.name}-{right_meta.name} on {attr}: {e}"
                    )
                    intermediate_meta = None
                    break

    elif granularity == "attribute":
        # Handle "attribute-at-a-time": each stage is an independent multi-way join on base data
        execution_model = "attribute-at-a-time"
        attribute_stage_specs = cast(List[MultiwayJoinStage], stages_spec)
        for stage_spec in attribute_stage_specs:
            try:
                analysis = _compute_attribute_stage_selectivity(
                    relation_metadata,
                    stage_spec.joining_attribute,
                    stage_spec.relations,
                )

                stage_analysis.append(analysis)
            except Exception as e:
                plan_logger.warning(
                    f"Could not compute attribute stage selectivity for {stage_spec.joining_attribute}: {e}"
                )

    # Calculate aggregate metrics for the entire plan
    total_intermediate_size = sum(s.result_size for s in stage_analysis)

    if stage_analysis:
        # Collect all selectivities across all stages for summary stats
        all_selectivities = [
            val for stage in stage_analysis for val in stage.selectivities.values()
        ]
        avg_selectivity, max_selectivity, min_selectivity = (
            _calculate_aggregate_metrics(all_selectivities)
        )
    else:
        avg_selectivity, max_selectivity, min_selectivity = (0.0, 0.0, 0.0)

    return JoinPlanStats(
        plan_id=metadata.get("plan_id", 0),
        pattern=metadata.get("pattern", "unknown"),
        granularity=granularity,
        relations=relations_in_scope,
        stage_stats=stage_analysis,
        total_intermediate_size=total_intermediate_size,
        average_selectivity=avg_selectivity,
        max_selectivity=max_selectivity,
        min_selectivity=min_selectivity,
        execution_model=execution_model,
    )


def _format_plan_details(analysis: JoinPlanStats) -> List[str]:
    """Formats the analysis results of a single plan into a human-readable string"""
    lines = [
        f"PLAN {analysis.plan_id} ({analysis.pattern}, {analysis.granularity}) - {analysis.execution_model.upper()}",
        "-" * 70,
        f"Relations in Scope: {', '.join(analysis.relations)}",
        f"Average Selectivity: {analysis.average_selectivity:.4f}",
        f"Total Intermediate Size: {analysis.total_intermediate_size:,}\n",
    ]

    stage_analysis = cast(List[JoinStageStats], analysis.stage_stats)
    if not stage_analysis:
        return lines

    if analysis.execution_model == "table-at-a-time":
        lines.append(f"Sequentially Binary Join Stages ({len(stage_analysis)} stages):")
        for stage in stage_analysis:
            r1, r2 = stage.relations
            lines.append(f"  {r1} ⋈ {r2} on {stage.join_attribute}:")
            lines.append(
                f"    Table sizes: {stage.relation_sizes[r1]:,} × {stage.relation_sizes[r2]:,}"
            )
            lines.append(
                f"    Distinct values: {stage.distinct_values_per_relation[r1]} ∩ {stage.distinct_values_per_relation[r2]} = {stage.common_values_across_all}"
            )
            lines.append(
                f"    Selectivities: {stage.selectivities[r1]:.4f}, {stage.selectivities[r2]:.4f}"
            )
            lines.append(f"    Result Size: {stage.result_size:,}")
            lines.append(f"    Reduction factor: {stage.reduction_factor:.6f}\n")

    elif analysis.execution_model == "attribute-at-a-time":
        lines.append(f"Multi-way Join Stages ({len(stage_analysis)} attributes):")
        for stage in stage_analysis:
            lines.append(f"  Multi-way join on {stage.join_attribute}:")
            lines.append(f"    Participating relations: {', '.join(stage.relations)}")
            lines.append(
                f"    Relation sizes: {', '.join([f'{r}:{s:,}' for r, s in stage.relation_sizes.items()])}"
            )
            lines.append(
                f"    Common values across all relations: {stage.common_values_across_all}"
            )
            lines.append(
                f"    Per-relation selectivities: {', '.join([f'{r}:{s:.4f}' for r, s in stage.selectivities.items()])}"
            )
            lines.append(f"    Result Size: {stage.result_size:,}")
            lines.append(f"    Reduction factor: {stage.reduction_factor:.6f}\n")
    return lines


def _generate_analysis_report(
    analyses: List[JoinPlanStats], output_path: pathlib.Path
) -> None:
    """Generates a comprehensive analysis report from a list of plan analyses"""

    report_path = output_path / "selectivity_analysis_report.txt"
    json_path = output_path / "selectivity_analysis_data.json"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("JOIN SELECTIVITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Plans Analyzed: {len(analyses)}\n\n")

        # Separate analyses by execution model for model-specific stats
        table_analyses = [a for a in analyses if a.execution_model == "table-at-a-time"]
        attribute_analyses = [
            a for a in analyses if a.execution_model == "attribute-at-a-time"
        ]

        if table_analyses or attribute_analyses:
            # --- Overall Statistics Section (high-level summary across all plans) ---
            f.write("OVERALL STATISTICS BY EXECUTION MODEL:\n")
            f.write("-" * 50 + "\n")

            if table_analyses:
                # Flatten selectivities from all table-based plans for aggregate stats
                table_selectivities = [
                    val
                    for a in table_analyses
                    for stage in cast(List[JoinStageStats], a.stage_stats)
                    for val in stage.selectivities.values()
                ]

                if table_selectivities:
                    f.write(f"Table-at-a-Time Plans ({len(table_analyses)} plans):\n")
                    f.write(
                        f"  Average Selectivity: {np.mean(table_selectivities):.4f}\n"
                    )
                    f.write(
                        f"  Median Selectivity: {np.median(table_selectivities):.4f}\n"
                    )
                    f.write(f"  Std Dev: {np.std(table_selectivities):.4f}\n\n")

            if attribute_analyses:
                # Flatten selectivities from all attribute-based plans
                attr_selectivities = [
                    val
                    for a in attribute_analyses
                    for stage in cast(List[JoinStageStats], a.stage_stats)
                    for val in stage.selectivities.values()
                ]

                if attr_selectivities:
                    f.write(
                        f"Attribute-at-a-Time Plans ({len(attribute_analyses)} plans):\n"
                    )
                    f.write(
                        f"  Average Selectivity: {np.mean(attr_selectivities):.4f}\n"
                    )
                    f.write(
                        f"  Median Selectivity: {np.median(attr_selectivities):.4f}\n"
                    )
                    f.write(f"  Std Dev: {np.std(attr_selectivities):.4f}\n\n")

        # --- Detailed Plan-by-Plan Section (breakdown for each plan) ---
        for analysis in analyses:
            details = _format_plan_details(analysis)
            f.write("\n".join(details))

    # Generate a single JSON file with all raw analysis data in a machine-readable format
    analysis_dicts = [asdict(analysis) for analysis in analyses]
    json_data_converted = _convert_numpy_types(analysis_dicts)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data_converted, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis report written to: {report_path}")
    logger.info(f"Analysis data written to: {json_path}")


def aggregate_individual_analyses(
    analysis_output_path: pathlib.Path,
) -> None:
    """Finds all individual analysis JSONs and aggregates them into a summary report"""
    logger.info("Aggregating individual analysis JSON files...")

    analysis_files = list(analysis_output_path.glob("analysis_plan_*.json"))

    if not analysis_files:
        logger.warning("No individual analysis JSON files found to aggregate")
        return

    logger.info(f"Found {len(analysis_files)} individual analysis files to aggregate")

    analyses = []
    for analysis_file in sorted(analysis_files):
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Rehydrate nested stage statistics from dicts back into JoinStageStats objects
                stage_data = data.get("stage_stats", [])
                data["stage_stats"] = [JoinStageStats(**s) for s in stage_data]
                analysis = JoinPlanStats(**data)
                analyses.append(analysis)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                f"Could not process individual analysis file {analysis_file.name}: {e}"
            )
    if analyses:
        _generate_analysis_report(analyses, analysis_output_path)
        logger.info(
            f"Aggregated {len(analyses)} individual analyses into comprehensive reports"
        )
    else:
        logger.warning("No valid analyses found to aggregate")
