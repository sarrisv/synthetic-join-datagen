# analysis_module.py
"""
Provides analysis capabilities for generated data and join plans,
including selectivity computation, join size estimation, and reporting.
"""

import pathlib
import logging
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class JoinSpec:
    """Specification for a single join operation."""
    table1: str
    table2: str
    join_attribute: str
    join_type: str = "inner"  # For future extension


@dataclass
class JoinSelectivity:
    """Results of selectivity analysis for a join."""
    table1: str
    table2: str
    join_attribute: str

    # Basic statistics
    table1_size: int
    table2_size: int
    table1_distinct_values: int
    table2_distinct_values: int
    common_values: int

    # Selectivity metrics
    table1_selectivity: float  # Fraction of table1 tuples that will participate
    table2_selectivity: float  # Fraction of table2 tuples that will participate
    estimated_join_size: int   # Estimated number of result tuples
    reduction_factor: float    # Join size / (table1_size * table2_size)

    # Distribution properties
    table1_null_count: int
    table2_null_count: int


@dataclass
class AttributeStageAnalysis:
    """Analysis for attribute-at-a-time multi-way join stage."""
    join_attribute: str
    participating_relations: List[str]
    relation_sizes: Dict[str, int]
    distinct_values_per_relation: Dict[str, int]
    common_values_across_all: int
    selectivities: Dict[str, float]  # Per relation
    estimated_result_size: int
    reduction_factor: float


@dataclass
class PlanAnalysis:
    """Complete analysis results for a single execution plan."""
    plan_id: int
    pattern: str
    granularity: str
    relations_in_scope: List[str]
    joins: List[JoinSelectivity]  # For table granularity
    attribute_stages: List[AttributeStageAnalysis]  # For attribute granularity

    # Aggregate metrics
    total_estimated_intermediate_size: int
    average_selectivity: float
    max_selectivity: float
    min_selectivity: float
    execution_model: str  # "table-at-a-time" or "attribute-at-a-time"


def load_relation_data(data_path: pathlib.Path, relation_name: str, format_name: str) -> pd.DataFrame:
    """Load relation data from file."""
    if format_name == "csv":
        file_path = data_path / f"{relation_name}.csv"
        return pd.read_csv(file_path)
    elif format_name == "json":
        file_path = data_path / f"{relation_name}.json"
        return pd.read_json(file_path, orient="records", lines=False)
    elif format_name == "parquet":
        dir_path = data_path / f"{relation_name}.parquet"
        return pd.read_parquet(dir_path)
    else:
        raise ValueError(f"Unsupported data format: {format_name}")


def parse_plan_file(plan_path: pathlib.Path) -> Tuple[Dict[str, str], List[JoinSpec]]:
    """Parse a plan file to extract metadata and join specifications."""
    metadata = {}
    joins = []

    with open(plan_path, 'r') as f:
        content = f.read()

    lines = content.strip().split('\n')

    # Parse metadata from header comments
    for line in lines:
        line = line.strip()
        if line.startswith('# Execution Plan'):
            metadata['plan_id'] = line.split()[-1]
        elif line.startswith('# Pattern for Fundamental Joins:'):
            metadata['pattern'] = line.split(': ', 1)[1]
        elif line.startswith('# Output Granularity:'):
            metadata['granularity'] = line.split(': ', 1)[1]
        elif line.startswith('# Relations in Plan Scope:'):
            scope_part = line.split(': ', 1)[1]
            if scope_part.strip() == 'None':
                metadata['relations_in_scope'] = []
            else:
                metadata['relations_in_scope'] = [r.strip() for r in scope_part.split(',')]

    # Parse join specifications
    parsing_joins = False
    for line in lines:
        line = line.strip()

        if 'Stages' in line and ('Binary Joins' in line or 'Multi-way Joins' in line):
            parsing_joins = True
            continue

        if parsing_joins and line and not line.startswith('#'):
            # Parse join line - format depends on granularity
            if metadata.get('granularity') == 'table':
                # Format: R0,R2,attr3
                parts = line.split(',')
                if len(parts) >= 3:
                    table1, table2 = parts[0], parts[1]
                    # Multiple join attributes possible
                    for attr in parts[2:]:
                        joins.append(JoinSpec(table1, table2, attr.strip()))
            elif metadata.get('granularity') == 'attribute':
                # Format: attr1,R0,R1,R2 (attribute followed by participating relations)
                parts = line.split(',')
                if len(parts) >= 3:
                    join_attr = parts[0]
                    relations = [r.strip() for r in parts[1:]]
                    # Create pairwise joins for all relations in this attribute stage
                    for i in range(len(relations)):
                        for j in range(i + 1, len(relations)):
                            joins.append(JoinSpec(relations[i], relations[j], join_attr))

    return metadata, joins


def compute_attribute_stage_selectivity(
    relation_data: Dict[str, pd.DataFrame],
    join_attribute: str,
    participating_relations: List[str]
) -> AttributeStageAnalysis:
    """Compute selectivity metrics for attribute-at-a-time multi-way join."""

    # Get attribute values from all participating relations
    relation_values = {}
    relation_sizes = {}
    distinct_values_per_relation = {}

    for rel_name in participating_relations:
        if rel_name in relation_data:
            rel_data = relation_data[rel_name]
            relation_sizes[rel_name] = len(rel_data)

            if join_attribute in rel_data.columns:
                values = rel_data[join_attribute].dropna()
                relation_values[rel_name] = set(values)
                distinct_values_per_relation[rel_name] = len(set(values))
            else:
                relation_values[rel_name] = set()
                distinct_values_per_relation[rel_name] = 0

    # Find values common across ALL participating relations
    if relation_values:
        common_values = set.intersection(*relation_values.values()) if relation_values else set()
    else:
        common_values = set()

    common_values_count = len(common_values)

    # Compute selectivities for each relation
    selectivities = {}
    for rel_name in participating_relations:
        if rel_name in relation_data and relation_sizes[rel_name] > 0:
            rel_data = relation_data[rel_name]
            if join_attribute in rel_data.columns:
                participating_tuples = rel_data[join_attribute].isin(common_values).sum()
                selectivities[rel_name] = participating_tuples / relation_sizes[rel_name]
            else:
                selectivities[rel_name] = 0.0
        else:
            selectivities[rel_name] = 0.0

    # Estimate result size for multi-way join
    # Use worst-case optimal bound: min over relations of |R| * |common_values|
    estimated_result_size = 0
    if common_values_count > 0:
        min_contribution = float('inf')
        for rel_name in participating_relations:
            if rel_name in relation_data:
                rel_data = relation_data[rel_name]
                if join_attribute in rel_data.columns:
                    contribution = 0
                    for value in common_values:
                        count = (rel_data[join_attribute] == value).sum()
                        contribution += count
                    min_contribution = min(min_contribution, contribution)

        if min_contribution != float('inf'):
            estimated_result_size = min_contribution

    # Reduction factor compared to Cartesian product
    cartesian_size = 1
    for rel_name in participating_relations:
        cartesian_size *= relation_sizes.get(rel_name, 1)

    reduction_factor = estimated_result_size / cartesian_size if cartesian_size > 0 else 0.0

    return AttributeStageAnalysis(
        join_attribute=join_attribute,
        participating_relations=participating_relations,
        relation_sizes=relation_sizes,
        distinct_values_per_relation=distinct_values_per_relation,
        common_values_across_all=common_values_count,
        selectivities=selectivities,
        estimated_result_size=estimated_result_size,
        reduction_factor=reduction_factor
    )


def compute_join_selectivity(
    table1_data: pd.DataFrame,
    table2_data: pd.DataFrame,
    join_spec: JoinSpec
) -> JoinSelectivity:
    """Compute selectivity metrics for a specific join."""

    attr = join_spec.join_attribute

    # Basic sizes
    table1_size = len(table1_data)
    table2_size = len(table2_data)

    # Get join attribute values (excluding nulls for join analysis)
    table1_values = table1_data[attr].dropna()
    table2_values = table2_data[attr].dropna()

    # Count nulls
    table1_null_count = table1_data[attr].isnull().sum()
    table2_null_count = table2_data[attr].isnull().sum()

    # Distinct values
    table1_distinct = set(table1_values)
    table2_distinct = set(table2_values)
    common_values_set = table1_distinct.intersection(table2_distinct)

    table1_distinct_count = len(table1_distinct)
    table2_distinct_count = len(table2_distinct)
    common_values_count = len(common_values_set)

    # Selectivity calculation
    # For each table, selectivity = fraction of tuples that have join attribute values
    # that exist in the other table
    if table1_size > 0:
        table1_participating = table1_values.isin(common_values_set).sum()
        table1_selectivity = table1_participating / table1_size
    else:
        table1_selectivity = 0.0

    if table2_size > 0:
        table2_participating = table2_values.isin(common_values_set).sum()
        table2_selectivity = table2_participating / table2_size
    else:
        table2_selectivity = 0.0

    # Estimate join size using simple heuristic
    # For each common value, multiply the counts from both tables
    estimated_join_size = 0
    for value in common_values_set:
        count1 = (table1_values == value).sum()
        count2 = (table2_values == value).sum()
        estimated_join_size += count1 * count2

    # Reduction factor
    cartesian_size = table1_size * table2_size
    reduction_factor = estimated_join_size / cartesian_size if cartesian_size > 0 else 0.0

    return JoinSelectivity(
        table1=join_spec.table1,
        table2=join_spec.table2,
        join_attribute=join_spec.join_attribute,
        table1_size=table1_size,
        table2_size=table2_size,
        table1_distinct_values=table1_distinct_count,
        table2_distinct_values=table2_distinct_count,
        common_values=common_values_count,
        table1_selectivity=table1_selectivity,
        table2_selectivity=table2_selectivity,
        estimated_join_size=estimated_join_size,
        reduction_factor=reduction_factor,
        table1_null_count=table1_null_count,
        table2_null_count=table2_null_count,
    )


def analyze_single_plan(
    plan_path: pathlib.Path,
    data_path: pathlib.Path,
    data_format: str
) -> PlanAnalysis:
    """Analyze a single execution plan file."""

    plan_logger = logging.getLogger(f"{__name__}.analyze_plan.{plan_path.stem}")
    plan_logger.debug(f"Analyzing plan: {plan_path.name}")

    # Parse the plan file
    metadata, join_specs = parse_plan_file(plan_path)
    granularity = metadata.get('granularity', 'unknown')

    # Load relation data
    relations_in_scope = metadata.get('relations_in_scope', [])
    relation_data = {}

    for relation in relations_in_scope:
        try:
            relation_data[relation] = load_relation_data(data_path, relation, data_format)
        except Exception as e:
            plan_logger.warning(f"Could not load data for relation {relation}: {e}")
            continue

    join_selectivities = []
    attribute_stages = []

    if granularity == 'table':
        # Table-at-a-time: analyze binary joins
        for join_spec in join_specs:
            if join_spec.table1 in relation_data and join_spec.table2 in relation_data:
                try:
                    selectivity = compute_join_selectivity(
                        relation_data[join_spec.table1],
                        relation_data[join_spec.table2],
                        join_spec
                    )
                    join_selectivities.append(selectivity)
                except Exception as e:
                    plan_logger.warning(f"Could not compute selectivity for join {join_spec.table1}-{join_spec.table2} on {join_spec.join_attribute}: {e}")
            else:
                plan_logger.warning(f"Missing data for join {join_spec.table1}-{join_spec.table2}")

    elif granularity == 'attribute':
        # Attribute-at-a-time: analyze multi-way joins per attribute
        # Group joins by attribute
        attribute_groups = defaultdict(set)
        for join_spec in join_specs:
            attribute_groups[join_spec.join_attribute].add(join_spec.table1)
            attribute_groups[join_spec.join_attribute].add(join_spec.table2)

        for attr, relations_set in attribute_groups.items():
            participating_relations = sorted(list(relations_set))
            try:
                stage_analysis = compute_attribute_stage_selectivity(
                    relation_data, attr, participating_relations
                )
                attribute_stages.append(stage_analysis)
            except Exception as e:
                plan_logger.warning(f"Could not compute attribute stage selectivity for {attr}: {e}")

    # Compute aggregate metrics based on granularity
    if granularity == 'table' and join_selectivities:
        selectivity_values = [
            (s.table1_selectivity + s.table2_selectivity) / 2
            for s in join_selectivities
        ]
        avg_selectivity = np.mean(selectivity_values)
        max_selectivity = np.max(selectivity_values)
        min_selectivity = np.min(selectivity_values)
        total_intermediate_size = sum(s.estimated_join_size for s in join_selectivities)
        execution_model = "table-at-a-time"
    elif granularity == 'attribute' and attribute_stages:
        # For attribute granularity, compute average selectivity across all relations in all stages
        all_selectivities = []
        for stage in attribute_stages:
            all_selectivities.extend(stage.selectivities.values())

        if all_selectivities:
            avg_selectivity = np.mean(all_selectivities)
            max_selectivity = np.max(all_selectivities)
            min_selectivity = np.min(all_selectivities)
        else:
            avg_selectivity = max_selectivity = min_selectivity = 0.0

        total_intermediate_size = sum(s.estimated_result_size for s in attribute_stages)
        execution_model = "attribute-at-a-time"
    else:
        avg_selectivity = max_selectivity = min_selectivity = 0.0
        total_intermediate_size = 0
        execution_model = "unknown"

    return PlanAnalysis(
        plan_id=int(metadata.get('plan_id', 0)),
        pattern=metadata.get('pattern', 'unknown'),
        granularity=granularity,
        relations_in_scope=relations_in_scope,
        joins=join_selectivities,
        attribute_stages=attribute_stages,
        total_estimated_intermediate_size=total_intermediate_size,
        average_selectivity=avg_selectivity,
        max_selectivity=max_selectivity,
        min_selectivity=min_selectivity,
        execution_model=execution_model,
    )


def generate_analysis_report(analyses: List[PlanAnalysis], output_path: pathlib.Path) -> None:
    """Generate a comprehensive analysis report."""

    report_path = output_path / "selectivity_analysis_report.txt"
    json_path = output_path / "selectivity_analysis_data.json"

    # Generate text report
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JOIN SELECTIVITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Plans Analyzed: {len(analyses)}\n\n")

        # Summary statistics by execution model
        table_analyses = [a for a in analyses if a.execution_model == "table-at-a-time"]
        attribute_analyses = [a for a in analyses if a.execution_model == "attribute-at-a-time"]

        if table_analyses or attribute_analyses:
            f.write("OVERALL STATISTICS BY EXECUTION MODEL:\n")
            f.write("-" * 50 + "\n")

            if table_analyses:
                table_selectivities = []
                for analysis in table_analyses:
                    table_selectivities.extend([
                        (s.table1_selectivity + s.table2_selectivity) / 2
                        for s in analysis.joins
                    ])

                if table_selectivities:
                    f.write(f"Table-at-a-Time Plans ({len(table_analyses)} plans):\n")
                    f.write(f"  Average Selectivity: {np.mean(table_selectivities):.4f}\n")
                    f.write(f"  Median Selectivity: {np.median(table_selectivities):.4f}\n")
                    f.write(f"  Std Dev: {np.std(table_selectivities):.4f}\n\n")

            if attribute_analyses:
                attr_selectivities = []
                for analysis in attribute_analyses:
                    for stage in analysis.attribute_stages:
                        attr_selectivities.extend(stage.selectivities.values())

                if attr_selectivities:
                    f.write(f"Attribute-at-a-Time Plans ({len(attribute_analyses)} plans):\n")
                    f.write(f"  Average Selectivity: {np.mean(attr_selectivities):.4f}\n")
                    f.write(f"  Median Selectivity: {np.median(attr_selectivities):.4f}\n")
                    f.write(f"  Std Dev: {np.std(attr_selectivities):.4f}\n\n")

        # Per-plan details
        for analysis in analyses:
            f.write(f"PLAN {analysis.plan_id} ({analysis.pattern}, {analysis.granularity}) - {analysis.execution_model.upper()}\\n")
            f.write("-" * 70 + "\\n")
            f.write(f"Relations in Scope: {', '.join(analysis.relations_in_scope)}\n")
            f.write(f"Average Selectivity: {analysis.average_selectivity:.4f}\n")
            f.write(f"Total Estimated Result Size: {analysis.total_estimated_intermediate_size:,}\n\n")

            if analysis.execution_model == "table-at-a-time" and analysis.joins:
                f.write(f"Binary Join Stages ({len(analysis.joins)} joins):\n")
                for join in analysis.joins:
                    f.write(f"  {join.table1} ⋈ {join.table2} on {join.join_attribute}:\n")
                    f.write(f"    Table sizes: {join.table1_size:,} × {join.table2_size:,}\n")
                    f.write(f"    Distinct values: {join.table1_distinct_values} ∩ {join.table2_distinct_values} = {join.common_values}\n")
                    f.write(f"    Selectivities: {join.table1_selectivity:.4f}, {join.table2_selectivity:.4f}\n")
                    f.write(f"    Estimated join size: {join.estimated_join_size:,}\n")
                    f.write(f"    Reduction factor: {join.reduction_factor:.6f}\n\n")

            elif analysis.execution_model == "attribute-at-a-time" and analysis.attribute_stages:
                f.write(f"Multi-way Join Stages ({len(analysis.attribute_stages)} attributes):\n")
                for stage in analysis.attribute_stages:
                    f.write(f"  Multi-way join on {stage.join_attribute}:\n")
                    f.write(f"    Participating relations: {', '.join(stage.participating_relations)}\n")
                    f.write(f"    Relation sizes: {', '.join([f'{r}:{s:,}' for r, s in stage.relation_sizes.items()])}\n")
                    f.write(f"    Common values across all relations: {stage.common_values_across_all}\n")
                    f.write(f"    Per-relation selectivities: {', '.join([f'{r}:{s:.4f}' for r, s in stage.selectivities.items()])}\n")
                    f.write(f"    Estimated result size: {stage.estimated_result_size:,}\n")
                    f.write(f"    Reduction factor: {stage.reduction_factor:.6f}\n\n")

            f.write("\n")

    # Generate JSON data
    json_data = []
    for analysis in analyses:
        plan_data = {
            "plan_id": analysis.plan_id,
            "pattern": analysis.pattern,
            "granularity": analysis.granularity,
            "execution_model": analysis.execution_model,
            "relations_in_scope": analysis.relations_in_scope,
            "aggregate_metrics": {
                "total_estimated_result_size": analysis.total_estimated_intermediate_size,
                "average_selectivity": analysis.average_selectivity,
                "max_selectivity": analysis.max_selectivity,
                "min_selectivity": analysis.min_selectivity,
            },
            "joins": [],
            "attribute_stages": []
        }

        # Add binary joins for table-at-a-time
        for join in analysis.joins:
            join_data = {
                "table1": join.table1,
                "table2": join.table2,
                "join_attribute": join.join_attribute,
                "table1_size": join.table1_size,
                "table2_size": join.table2_size,
                "table1_distinct_values": join.table1_distinct_values,
                "table2_distinct_values": join.table2_distinct_values,
                "common_values": join.common_values,
                "table1_selectivity": join.table1_selectivity,
                "table2_selectivity": join.table2_selectivity,
                "estimated_join_size": join.estimated_join_size,
                "reduction_factor": join.reduction_factor,
                "table1_null_count": join.table1_null_count,
                "table2_null_count": join.table2_null_count,
            }
            plan_data["joins"].append(join_data)

        # Add attribute stages for attribute-at-a-time
        for stage in analysis.attribute_stages:
            stage_data = {
                "join_attribute": stage.join_attribute,
                "participating_relations": stage.participating_relations,
                "relation_sizes": stage.relation_sizes,
                "distinct_values_per_relation": stage.distinct_values_per_relation,
                "common_values_across_all": stage.common_values_across_all,
                "selectivities": stage.selectivities,
                "estimated_result_size": stage.estimated_result_size,
                "reduction_factor": stage.reduction_factor,
            }
            plan_data["attribute_stages"].append(stage_data)

        json_data.append(plan_data)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    json_data_converted = convert_numpy_types(json_data)

    with open(json_path, 'w') as f:
        json.dump(json_data_converted, f, indent=2)

    logger.info(f"Analysis report written to: {report_path}")
    logger.info(f"Analysis data written to: {json_path}")


def generate_individual_analysis_report(analysis: PlanAnalysis, output_path: pathlib.Path) -> None:
    """Generate an individual analysis report for a single plan."""

    try:
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"INDIVIDUAL PLAN ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"PLAN {analysis.plan_id} ({analysis.pattern}, {analysis.granularity}) - {analysis.execution_model.upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Relations in Scope: {', '.join(analysis.relations_in_scope)}\n")
            f.write(f"Average Selectivity: {analysis.average_selectivity:.4f}\n")
            f.write(f"Total Estimated Result Size: {analysis.total_estimated_intermediate_size:,}\n\n")

            if analysis.execution_model == "table-at-a-time" and analysis.joins:
                f.write(f"Binary Join Stages ({len(analysis.joins)} joins):\n")
                for join in analysis.joins:
                    f.write(f"  {join.table1} ⋈ {join.table2} on {join.join_attribute}:\n")
                    f.write(f"    Table sizes: {join.table1_size:,} × {join.table2_size:,}\n")
                    f.write(f"    Distinct values: {join.table1_distinct_values} ∩ {join.table2_distinct_values} = {join.common_values}\n")
                    f.write(f"    Selectivities: {join.table1_selectivity:.4f}, {join.table2_selectivity:.4f}\n")
                    f.write(f"    Estimated join size: {join.estimated_join_size:,}\n")
                    f.write(f"    Reduction factor: {join.reduction_factor:.6f}\n\n")

            elif analysis.execution_model == "attribute-at-a-time" and analysis.attribute_stages:
                f.write(f"Multi-way Join Stages ({len(analysis.attribute_stages)} attributes):\n")
                for stage in analysis.attribute_stages:
                    f.write(f"  Multi-way join on {stage.join_attribute}:\n")
                    f.write(f"    Participating relations: {', '.join(stage.participating_relations)}\n")
                    f.write(f"    Relation sizes: {', '.join([f'{r}:{s:,}' for r, s in stage.relation_sizes.items()])}\n")
                    f.write(f"    Common values across all relations: {stage.common_values_across_all}\n")
                    f.write(f"    Per-relation selectivities: {', '.join([f'{r}:{s:.4f}' for r, s in stage.selectivities.items()])}\n")
                    f.write(f"    Estimated result size: {stage.estimated_result_size:,}\n")
                    f.write(f"    Reduction factor: {stage.reduction_factor:.6f}\n\n")

            f.write(f"\nGenerated at: {output_path}\n")

    except Exception as e:
        logger.error(f"Failed to write individual analysis report to {output_path}: {e}")


def analyze_all_plans(
    plans_path: pathlib.Path,
    data_path: pathlib.Path,
    data_format: str,
    output_base_path: pathlib.Path
) -> List[PlanAnalysis]:
    """Analyze all plan files in the plans directory."""

    logger.info(f"Analyzing plans in: {plans_path}")

    # Find all plan files (exclude DOT files)
    plan_files = []
    for file_path in plans_path.glob("plan_*.txt"):
        plan_files.append(file_path)

    if not plan_files:
        logger.warning(f"No plan files found in {plans_path}")
        return []

    logger.info(f"Found {len(plan_files)} plan files to analyze")

    # Analyze each plan
    analyses = []
    for plan_file in sorted(plan_files):
        try:
            analysis = analyze_single_plan(plan_file, data_path, data_format)
            analyses.append(analysis)
        except Exception as e:
            logger.error(f"Failed to analyze plan {plan_file.name}: {e}")

    # Generate report
    if analyses:
        generate_analysis_report(analyses, output_base_path)

    return analyses


def aggregate_individual_analyses(base_output_path: pathlib.Path, plans_path: pathlib.Path) -> None:
    """Aggregate individual analysis files into comprehensive reports."""

    logger.info("Aggregating individual analysis files...")

    # Find all individual analysis files
    analysis_files = list(base_output_path.glob("analysis_plan_*.txt"))

    if not analysis_files:
        logger.warning("No individual analysis files found to aggregate")
        return

    logger.info(f"Found {len(analysis_files)} individual analysis files")

    # Parse individual files to extract analysis data
    analyses = []
    for analysis_file in sorted(analysis_files):
        try:
            # Extract plan info from filename
            filename = analysis_file.stem
            # Format: analysis_plan_{id}_{pattern}_{granularity}
            parts = filename.split('_')
            if len(parts) >= 5:
                plan_id = int(parts[2])
                pattern = parts[3]
                granularity = parts[4]

                # Find corresponding plan file
                plan_file = plans_path / f"plan_{plan_id}_{pattern}_{granularity}.txt"
                if plan_file.exists():
                    # Re-analyze to get full data structure
                    data_path = base_output_path / "data"
                    analysis = analyze_single_plan(plan_file, data_path, "csv")
                    analyses.append(analysis)

        except Exception as e:
            logger.warning(f"Could not process individual analysis file {analysis_file.name}: {e}")

    # Generate comprehensive aggregated report
    if analyses:
        generate_analysis_report(analyses, base_output_path)
        logger.info(f"Aggregated {len(analyses)} individual analyses into comprehensive reports")
    else:
        logger.warning("No valid analyses found to aggregate")
