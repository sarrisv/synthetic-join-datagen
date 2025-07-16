import json
import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from itertools import chain
from typing import Any, Dict, List, Tuple, Union, cast
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class JoinSpec:
    """Specification for a single join operation."""

    table1: str
    table2: str
    join_attribute: str
    join_type: str = "inner"  # For future extension


@dataclass
class AttributeStageSpec:
    """Specification for an attribute-at-a-time multi-way join stage."""

    join_attribute: str
    participating_relations: List[str]



@dataclass
class JoinSelectivity:
    """Results of selectivity analysis for a join"""

    rel1: str
    rel2: str
    join_attribute: str

    # Basic statistics
    rel1_size: int
    rel2_size: int
    rel1_distinct_values: int
    rel2_distinct_values: int
    common_values: int

    # Selectivity metrics
    rel1_selectivity: float  # Fraction of rel1 tuples that will participate
    rel2_selectivity: float  # Fraction of rel2 tuples that will participate
    estimated_join_size: int  # Estimated number of result tuples
    reduction_factor: float  # Join size / (rel1_size * rel2_size)

    # Distribution properties
    rel1_null_count: int
    rel2_null_count: int


@dataclass
class AttributeStageAnalysis:
    """Analysis for attribute-at-a-time multi-way join stage"""

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
    """Complete analysis results for a single execution plan"""

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


def _convert_numpy_types(obj: Any) -> Any:
    """Helper to convert numpy types to native Python types for JSON serialization."""
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


def save_plan_analysis_to_json(
    analysis: PlanAnalysis, output_path: pathlib.Path
) -> None:
    """Saves a single PlanAnalysis object to a JSON file."""
    analysis_dict = asdict(analysis)
    analysis_dict_native = _convert_numpy_types(analysis_dict)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_dict_native, f, indent=2)
        logger.debug(f"Saved individual analysis to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save individual analysis JSON to {output_path}: {e}")


def _enforce_nullable_integers(df: pd.DataFrame) -> pd.DataFrame:
    """Converts attribute columns (except PK 'attr0') to nullable integer type."""
    cols_to_convert = {
        col: pd.Int64Dtype() for col in df.columns if col.startswith("attr") and col != "attr0"
    }
    if cols_to_convert:
        # Using .astype() on a subset of columns is safer
        return df.astype(cols_to_convert)
    return df


def load_relation_data(
    data_path: pathlib.Path, relation_name: str, format_name: str
) -> pd.DataFrame:
    """Load relation data from file."""
    if format_name == "csv":
        file_path = data_path / f"{relation_name}.csv"
        # Read header to build a dtype dictionary. This ensures that columns
        # with nulls are read as nullable integers (Int64) instead of float64.
        try:
            header = pd.read_csv(file_path, nrows=0).columns.tolist()
            dtypes = {col: pd.Int64Dtype() for col in header if col.startswith("attr")}
            if "attr0" in dtypes:
                dtypes["attr0"] = np.int64  # Primary key is non-nullable
            return pd.read_csv(file_path, dtype=dtypes, keep_default_na=True)
        except Exception as e:
            logger.warning(f"Could not read CSV with specified dtypes for {relation_name}, falling back. Error: {e}")
            # On fallback, still try to enforce the correct dtypes to avoid downstream issues.
            df = pd.read_csv(file_path)
            return _enforce_nullable_integers(df)
    elif format_name == "json":
        file_path = data_path / f"{relation_name}.json"
        df = pd.read_json(file_path, orient="records", lines=False)
        # Enforce nullable integer types post-loading for consistency.
        return _enforce_nullable_integers(df)
    elif format_name == "parquet":
        dir_path = data_path / f"{relation_name}.parquet"
        return pd.read_parquet(dir_path)
    else:
        raise ValueError(f"Unsupported data format: {format_name}")


def parse_plan_file(
    plan_path: pathlib.Path,
) -> Tuple[Dict[str, Any], Union[List[JoinSpec], List[AttributeStageSpec]]]:
    """Parse a plan file to extract metadata and join specifications."""
    metadata: Dict[str, Any] = {}
    table_joins: List[JoinSpec] = []
    attr_stages: List[AttributeStageSpec] = []

    with open(plan_path, "r") as f:
        content = f.read()

    lines = content.strip().split("\n")

    # Parse metadata from header comments
    for line in lines:
        line = line.strip()
        if line.startswith("# Execution Plan"):
            metadata["plan_id"] = int(line.split()[-1])
        elif line.startswith("# Pattern for Base Joins:"):
            metadata["pattern"] = line.split(": ", 1)[1]
        elif line.startswith("# Output Granularity:"):
            metadata["granularity"] = line.split(": ", 1)[1]
        elif line.startswith("# Relations in Plan Scope:"):
            scope_part = line.split(": ", 1)[1]
            if scope_part.strip() == "None":
                metadata["relations_in_scope"] = []
            else:
                metadata["relations_in_scope"] = [
                    r.strip() for r in scope_part.split(",") if r.strip()
                ]

    # Parse join specifications
    parsing_joins = False
    for line in lines:
        line = line.strip()

        if "Stages" in line and ("Binary Joins" in line or "Multi-way Joins" in line):
            parsing_joins = True
            continue

        if parsing_joins and line and not line.startswith("#"):
            # Parse join line - format depends on granularity
            if metadata.get("granularity") == "table":
                # Format: R0,R2,attr3
                parts = line.split(",")
                if len(parts) >= 3:
                    table1, table2 = parts[0], parts[1]
                    # Multiple join attributes possible
                    for attr in parts[2:]:
                        table_joins.append(JoinSpec(table1, table2, attr.strip()))
            elif metadata.get("granularity") == "attribute":
                # Format: attr1,R0,R1,R2 (attribute followed by participating relations)
                parts = line.split(",")
                if len(parts) >= 2:  # e.g., attr1,R0
                    join_attr = parts[0]
                    relations = [r.strip() for r in parts[1:]]
                    attr_stages.append(AttributeStageSpec(join_attr, relations))

    if metadata.get("granularity") == "attribute":
        return metadata, attr_stages
    return metadata, table_joins


def compute_attribute_stage_selectivity(
    relation_data: Dict[str, pd.DataFrame],
    join_attribute: str,
    participating_relations: List[str],
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
        common_values = (
            set.intersection(*relation_values.values()) if relation_values else set()
        )
    else:
        common_values = set()

    common_values_count = len(common_values)

    # Compute selectivities for each relation
    selectivities = {}
    for rel_name in participating_relations:
        if rel_name in relation_data and relation_sizes[rel_name] > 0:
            rel_data = relation_data[rel_name]
            if join_attribute in rel_data.columns:
                participating_tuples = (
                    rel_data[join_attribute].isin(common_values).sum()
                )
                selectivities[rel_name] = (
                    participating_tuples / relation_sizes[rel_name]
                )
            else:
                selectivities[rel_name] = 0.0
        else:
            selectivities[rel_name] = 0.0

    # Estimate the exact result size for the multi-way join.
    # This is done by calculating Sum_v( Product_i( count(v, Ri) ) ) over all common
    # values 'v' that exist in all participating relations 'Ri'.
    value_counts_per_relation = []
    # Check if all relations are available and have the join attribute
    all_relations_valid = True
    for rel_name in participating_relations:
        if not (rel_name in relation_data and join_attribute in relation_data[rel_name].columns):
            all_relations_valid = False
            break
        counts = relation_data[rel_name][join_attribute].dropna().value_counts()
        counts.name = rel_name
        value_counts_per_relation.append(counts)

    estimated_result_size = 0
    if all_relations_valid and value_counts_per_relation:
        # Concatenate all value count Series into a single DataFrame. The index will be the
        # union of all join key values across all relations. `axis=1` makes each Series a column.
        combined_counts_df = pd.concat(value_counts_per_relation, axis=1, sort=False)
        # The join result only includes tuples where the join key value exists in ALL relations.
        # `dropna()` filters the DataFrame to only keep rows (join key values) present everywhere.
        combined_counts_df.dropna(inplace=True)
        if not combined_counts_df.empty:
            # Calculate the product of counts for each common value (row-wise).
            product_of_counts = combined_counts_df.product(axis=1)
            # Sum these products to get the total exact join size.
            estimated_result_size = int(product_of_counts.sum())

    # Reduction factor compared to Cartesian product
    cartesian_size = 1
    for rel_name in participating_relations:
        cartesian_size *= relation_sizes.get(rel_name, 1)

    reduction_factor = (
        estimated_result_size / cartesian_size if cartesian_size > 0 else 0.0
    )

    return AttributeStageAnalysis(
        join_attribute=join_attribute,
        participating_relations=participating_relations,
        relation_sizes=relation_sizes,
        distinct_values_per_relation=distinct_values_per_relation,
        common_values_across_all=common_values_count,
        selectivities=selectivities,
        estimated_result_size=estimated_result_size,
        reduction_factor=reduction_factor,
    )


def compute_join_selectivity(
    table1_data: pd.DataFrame, table2_data: pd.DataFrame, join_spec: JoinSpec
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

    # Estimate exact join size by multiplying the counts of each common value
    # from both tables and summing the results.
    if common_values_set:
        # This vectorized approach is much faster than iterating.
        counts1 = table1_values.value_counts()
        counts2 = table2_values.value_counts()
        # Pandas aligns on the index (the values) automatically, and the sum ignores NaNs.
        estimated_join_size = int((counts1 * counts2).sum())
    else:
        estimated_join_size = 0
    # Reduction factor
    cartesian_size = table1_size * table2_size
    reduction_factor = (
        estimated_join_size / cartesian_size if cartesian_size > 0 else 0.0
    )

    return JoinSelectivity(
        rel1=join_spec.table1,
        rel2=join_spec.table2,
        join_attribute=join_spec.join_attribute,
        rel1_size=table1_size,
        rel2_size=table2_size,
        rel1_distinct_values=table1_distinct_count,
        rel2_distinct_values=table2_distinct_count,
        common_values=common_values_count,
        rel1_selectivity=table1_selectivity,
        rel2_selectivity=table2_selectivity,
        estimated_join_size=estimated_join_size,
        reduction_factor=reduction_factor,
        rel1_null_count=table1_null_count,
        rel2_null_count=table2_null_count,
    )


def analyze_single_plan(
    plan_path: pathlib.Path, data_path: pathlib.Path, data_format: str
) -> PlanAnalysis:
    """Analyze a single execution plan file."""

    plan_logger = logging.getLogger(f"{__name__}.analyze_plan.{plan_path.stem}")
    plan_logger.debug(f"Analyzing plan: {plan_path.name}")

    metadata, stages_spec = parse_plan_file(plan_path)
    granularity = metadata.get("granularity", "unknown")

    # Load relation data
    relations_in_scope = metadata.get("relations_in_scope", [])
    relation_data = {}

    for relation in relations_in_scope:
        try:
            relation_data[relation] = load_relation_data(
                data_path, relation, data_format
            )
        except Exception as e:
            plan_logger.warning(f"Could not load data for relation {relation}: {e}")
            continue

    join_selectivities = []
    attribute_stages = []

    if granularity == "table":
        # Table-at-a-time: analyze binary joins
        join_specs = cast(List[JoinSpec], stages_spec)
        for join_spec in join_specs:
            if join_spec.table1 in relation_data and join_spec.table2 in relation_data:
                try:
                    selectivity = compute_join_selectivity(
                        relation_data[join_spec.table1],
                        relation_data[join_spec.table2],
                        join_spec,
                    )
                    join_selectivities.append(selectivity)
                except Exception as e:
                    plan_logger.warning(
                        f"Could not compute selectivity for join {join_spec.table1}-{join_spec.table2} on {join_spec.join_attribute}: {e}"
                    )
            else:
                plan_logger.warning(
                    f"Missing data for join {join_spec.table1}-{join_spec.table2}"
                )

    elif granularity == "attribute":
        # Attribute-at-a-time: analyze multi-way joins per attribute
        attribute_stage_specs = cast(List[AttributeStageSpec], stages_spec)
        for stage_spec in attribute_stage_specs:
            try:
                stage_analysis = compute_attribute_stage_selectivity(
                    relation_data,
                    stage_spec.join_attribute,
                    stage_spec.participating_relations,
                )
                attribute_stages.append(stage_analysis)
            except Exception as e:
                plan_logger.warning(
                    f"Could not compute attribute stage selectivity for {stage_spec.join_attribute}: {e}"
                )

    # Compute aggregate metrics based on granularity
    if granularity == "table" and join_selectivities:
        selectivity_values = [
            (s.rel1_selectivity + s.rel2_selectivity) / 2 for s in join_selectivities
        ]
        avg_selectivity = np.mean(selectivity_values)
        max_selectivity = np.max(selectivity_values)
        min_selectivity = np.min(selectivity_values)
        total_intermediate_size = sum(s.estimated_join_size for s in join_selectivities)
        execution_model = "table-at-a-time"
    elif granularity == "attribute" and attribute_stages:
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
        plan_id=metadata.get("plan_id", 0),
        pattern=metadata.get("pattern", "unknown"),
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


def _format_plan_details(analysis: PlanAnalysis) -> List[str]:
    """Helper function to format the detailed part of a plan analysis into a list of strings."""
    lines = []
    lines.append(
        f"PLAN {analysis.plan_id} ({analysis.pattern}, {analysis.granularity}) - {analysis.execution_model.upper()}"
    )
    lines.append("-" * 70)
    lines.append(f"Relations in Scope: {', '.join(analysis.relations_in_scope)}")
    lines.append(f"Average Selectivity: {analysis.average_selectivity:.4f}")
    lines.append(
        f"Total Estimated Result Size: {analysis.total_estimated_intermediate_size:,}\n"
    )

    if analysis.execution_model == "table-at-a-time" and analysis.joins:
        lines.append(f"Binary Join Stages ({len(analysis.joins)} joins):")
        for join in analysis.joins:
            lines.append(f"  {join.rel1} ⋈ {join.rel2} on {join.join_attribute}:")
            lines.append(f"    Table sizes: {join.rel1_size:,} × {join.rel2_size:,}")
            lines.append(
                f"    Distinct values: {join.rel1_distinct_values} ∩ {join.rel2_distinct_values} = {join.common_values}"
            )
            lines.append(
                f"    Selectivities: {join.rel1_selectivity:.4f}, {join.rel2_selectivity:.4f}"
            )
            lines.append(f"    Estimated join size: {join.estimated_join_size:,}")
            lines.append(f"    Reduction factor: {join.reduction_factor:.6f}\n")

    elif (
        analysis.execution_model == "attribute-at-a-time" and analysis.attribute_stages
    ):
        lines.append(
            f"Multi-way Join Stages ({len(analysis.attribute_stages)} attributes):"
        )
        for stage in analysis.attribute_stages:
            lines.append(f"  Multi-way join on {stage.join_attribute}:")
            lines.append(
                f"    Participating relations: {', '.join(stage.participating_relations)}"
            )
            lines.append(
                f"    Relation sizes: {', '.join([f'{r}:{s:,}' for r, s in stage.relation_sizes.items()])}"
            )
            lines.append(
                f"    Common values across all relations: {stage.common_values_across_all}"
            )
            lines.append(
                f"    Per-relation selectivities: {', '.join([f'{r}:{s:.4f}' for r, s in stage.selectivities.items()])}"
            )
            lines.append(f"    Estimated result size: {stage.estimated_result_size:,}")
            lines.append(f"    Reduction factor: {stage.reduction_factor:.6f}\n")
    return lines


def generate_analysis_report(
    analyses: List[PlanAnalysis], output_path: pathlib.Path
) -> None:
    """Generate a comprehensive analysis report."""

    report_path = output_path / "selectivity_analysis_report.txt"
    json_path = output_path / "selectivity_analysis_data.json"

    # Generate text report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("JOIN SELECTIVITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Plans Analyzed: {len(analyses)}\n\n")

        # Summary statistics by execution model
        table_analyses = [a for a in analyses if a.execution_model == "table-at-a-time"]
        attribute_analyses = [
            a for a in analyses if a.execution_model == "attribute-at-a-time"
        ]

        if table_analyses or attribute_analyses:
            f.write("OVERALL STATISTICS BY EXECUTION MODEL:\n")
            f.write("-" * 50 + "\n")

            if table_analyses:
                # Use a more direct and efficient way to gather selectivities
                table_selectivities = [
                    (s.rel1_selectivity + s.rel2_selectivity) / 2
                    for s in chain.from_iterable(a.joins for a in table_analyses)
                ]

                if table_selectivities:
                    f.write(f"Table-at-a-Time Plans ({len(table_analyses)} plans):\n")
                    f.write(f"  Average Selectivity: {np.mean(table_selectivities):.4f}\n")
                    f.write(f"  Median Selectivity: {np.median(table_selectivities):.4f}\n")
                    f.write(f"  Std Dev: {np.std(table_selectivities):.4f}\n\n")

            if attribute_analyses:
                attr_selectivities = [
                    val
                    for stage in chain.from_iterable(
                        a.attribute_stages for a in attribute_analyses
                    )
                    for val in stage.selectivities.values()
                ]

                if attr_selectivities:
                    f.write(f"Attribute-at-a-Time Plans ({len(attribute_analyses)} plans):\n")
                    f.write(f"  Average Selectivity: {np.mean(attr_selectivities):.4f}\n")
                    f.write(f"  Median Selectivity: {np.median(attr_selectivities):.4f}\n")
                    f.write(f"  Std Dev: {np.std(attr_selectivities):.4f}\n\n")

        # Per-plan details
        for analysis in analyses:
            details = _format_plan_details(analysis)
            f.write("\n".join(details))

    # Generate JSON data using dataclasses.asdict for simplicity and robustness
    analysis_dicts = [asdict(analysis) for analysis in analyses]
    json_data_converted = _convert_numpy_types(analysis_dicts)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data_converted, f, indent=2)

    logger.info(f"Analysis report written to: {report_path}")
    logger.info(f"Analysis data written to: {json_path}")


def aggregate_individual_analyses(
    analysis_output_path: pathlib.Path,
) -> None:
    """Aggregate individual analysis JSON files into a comprehensive report."""

    logger.info("Aggregating individual analysis JSON files...")

    # Find all individual analysis JSON files
    analysis_files = list(analysis_output_path.glob("analysis_plan_*.json"))

    if not analysis_files:
        logger.warning("No individual analysis JSON files found to aggregate.")
        return

    logger.info(f"Found {len(analysis_files)} individual analysis files to aggregate.")

    analyses = []
    for analysis_file in sorted(analysis_files):
        try:
            with open(analysis_file, "r", encoding="utf-8") as f:
                data = json.load(f)

                joins = [JoinSelectivity(**j) for j in data.get("joins", [])]
                attr_stages = [
                    AttributeStageAnalysis(**s)
                    for s in data.get("attribute_stages", [])
                ]

                analysis = PlanAnalysis(
                    plan_id=data["plan_id"],
                    pattern=data["pattern"],
                    granularity=data["granularity"],
                    relations_in_scope=data["relations_in_scope"],
                    joins=joins,
                    attribute_stages=attr_stages,
                    total_estimated_intermediate_size=data[
                        "total_estimated_intermediate_size"
                    ],
                    average_selectivity=data["average_selectivity"],
                    max_selectivity=data["max_selectivity"],
                    min_selectivity=data["min_selectivity"],
                    execution_model=data["execution_model"],
                )
                analyses.append(analysis)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(
                f"Could not process individual analysis file {analysis_file.name}: {e}"
            )
    if analyses:
        generate_analysis_report(analyses, analysis_output_path)
        logger.info(
            f"Aggregated {len(analyses)} individual analyses into comprehensive reports."
        )
    else:
        logger.warning("No valid analyses found to aggregate.")
