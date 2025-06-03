# Synthetic Join Data Generator

A research tool for generating synthetic database relations and join execution plans.

## Overview

This project generates:
- **Synthetic Relations**: Configurable datasets with various data distributions, null values, and domain constraints
- **Join Execution Plans**: Fundamental join plans using different join patterns (random, star, chain, cyclic) with table-granularity and attribute-granularity outputs
- **Selectivity Analysis**: Comprehensive analysis of join selectivities, estimated result sizes, and reduction factors

Originally developed for academic research comparing table-at-a-time binary joins vs. worst-case optimal join algorithms.

## Features

### Data Generation
- **Multiple distributions**: Uniform, Zipf, Normal
- **Configurable parameters**: Domain size, null percentages, duplication factors
- **Scalable generation**: Uses Dask for parallel processing
- **Multiple output formats**: CSV, JSON, Parquet
- **Reproducible**: Seed-based generation for consistent results

### Plan Generation
- **Join patterns**: Random, Star, Chain, Cyclic join patterns
- **Multiple granularities**: Table-at-a-time vs. attribute-at-a-time plans
- **Visualization**: DOT graph generation for join patterns
- **Flexible output**: Text and JSON plan formats

### Selectivity Analysis ✨ NEW
- **Join selectivity computation**: Analyzes actual data to compute join selectivities
- **Result size estimation**: Estimates intermediate and final join result sizes
- **Comprehensive reporting**: Text and JSON analysis reports
- **Multiple metrics**: Reduction factors, distinct value overlaps, null handling

### Configuration File Support ✨ NEW
- **YAML configuration**: Use config files instead of long CLI commands
- **CLI override**: Command-line arguments take precedence over config values
- **Template configs**: Example configurations for common use cases

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DaskV2

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Quick Start

### Using Command Line
```bash
# Generate 5 relations with 4 attributes, 10K tuples each, and 2 plans per pattern
python main.py --relations 5 --attributes 4 --unique-tuples 10000 \
               --plans 2 --join-pattern random star \
               --run-analysis

# Generate with Zipf distribution and high skew
python main.py --distribution zipf --skew 3.0 --domain-size 1000 \
               --null-percentage 0.1 --run-analysis
```

### Using Configuration Files
```bash
# Use a configuration file
python main.py --config example_config.yaml

# Override specific values from config
python main.py --config example_config.yaml --relations 10 --verbose
```

### Example Configuration File
```yaml
# experiment_config.yaml
relations: 5
attributes: 4
unique-tuples: 10000
distribution: "zipf"
skew: 2.5
domain-size: 1000
null-percentage: 0.05

plans: 3
join-pattern: [\"random\", \"star\", \"chain\"]
plan-granularity: [\"table\", \"attribute\"]
generate-dot-visualization: true

run-analysis: true
verbose: true
```

## Command Line Options

### Data Generation
- `--relations N`: Number of relations to generate (default: 3)
- `--attributes M`: Number of attributes per relation (default: 3)
- `--unique-tuples U`: Unique tuples per relation (default: 1000)
- `--duplication-factor D`: Tuple duplication factor (default: 1)
- `--distribution {uniform,zipf,normal}`: Value distribution (default: normal)
- `--skew S`: Skew parameter for Zipf distribution (default: 2.0)
- `--domain-size MAX`: Maximum value for non-primary-key attributes (default: 1000)
- `--null-percentage P`: Percentage of null values [0.0, 1.0) (default: 0.0)
- `--data-output-format {csv,json,parquet}`: Data file format (default: csv)

### Plan Generation
- `--plans N`: Number of fundamental plans per pattern (default: 1)
- `--join-pattern PATTERNS`: Join patterns to use (default: random)
- `--plan-granularity {table,attribute}`: Plan output granularity (default: table)
- `--max-join-relations MAX`: Maximum relations in plan scope (default: all)
- `--plan-output-format {txt,json}`: Plan file format (default: txt)
- `--generate-dot-visualization`: Generate DOT graph files

### Analysis
- `--run-analysis`: Run selectivity analysis on generated data and plans

### Configuration
- `--config CONFIG_FILE`: Path to YAML configuration file


### System
- `--seed S`: Global random seed for reproducibility
- `--base-output-dir PATH`: Base output directory (default: generated_output_dask)
- `--dask-workers N`: Number of Dask workers
- `--dask-threads-per-worker N`: Threads per Dask worker
- `--verbose`: Enable detailed logging

## Output Structure

```
generated_output/
├── data/                          # Generated relation files
│   ├── R0.csv
│   ├── R1.csv
│   └── ...
├── plans/                         # Generated plan files
│   ├── plan_0_random_table.txt
│   ├── plan_1_star_attribute.json
│   ├── plan_2_chain_table_viz.dot
│   └── ...
├── selectivity_analysis_report.txt   # Human-readable analysis report
└── selectivity_analysis_data.json    # Machine-readable analysis data
```

## Selectivity Analysis Output

The analysis feature generates comprehensive reports including:

### Text Report (`selectivity_analysis_report.txt`)
```
PLAN 0 (random, table)
Relations in Scope: R0, R1, R2
Number of Joins: 2
Average Selectivity: 0.8902

Join Details:
  R0 ⋈ R1 on attr1:
    Table sizes: 1,000 × 1,000
    Distinct values: 556 ∩ 550 = 492
    Selectivities: 0.9260, 0.9380
    Estimated join size: 2,204
    Reduction factor: 0.002204
```

### JSON Data (`selectivity_analysis_data.json`)
```json
{
  "plan_id": 0,
  "pattern": "random",
  "granularity": "table",
  "joins": [
    {
      "table1": "R0",
      "table2": "R1",
      "join_attribute": "attr1",
      "table1_selectivity": 0.926,
      "table2_selectivity": 0.938,
      "estimated_join_size": 2204,
      "reduction_factor": 0.002204
    }
  ]
}
```

## Research Applications

This tool is designed for:
- **Join Algorithm Evaluation**: Compare performance of different join algorithms
- **Query Optimization Research**: Test plan selection and cost estimation strategies
- **Cardinality Estimation**: Evaluate estimation accuracy across data distributions
- **Distributed Join Processing**: Test join patterns across different partitioning schemes
- **Selectivity Analysis**: Understand join behavior across different data patterns

## Architecture

- **`main.py`**: CLI orchestration and Dask cluster management
- **`data_generation_module.py`**: Parallel relation generation using Dask
- **`plan_generation_module.py`**: Join plan generation and DOT visualization
- **`analysis_module.py`**: Selectivity analysis and reporting ✨ NEW
- **`registries.py`**: Extensible pattern implementations for distributions, join patterns, and formatters

## Extending the Tool

### Adding New Distributions
```python
class CustomDistribution(DistributionStrategy):
    def generate_numpy_column(self, rng_numpy, num_values, domain_size, skew=None):
        # Your custom distribution logic
        return rng_numpy.custom_distribution(...)

# Register in registries.py
DISTRIBUTION_STRATEGIES["custom"] = CustomDistribution
```

### Adding New Join Patterns
```python
class CustomJoinPattern(JoinPattern):
    def generate_fundamental_joins(self, relations, num_attributes, py_random):
        # Your join pattern logic
        return [(rel1, rel2, attr_idx), ...]

# Register in registries.py
JOIN_PATTERNS["custom"] = CustomJoinPattern
```

## Performance Notes

- Uses Dask for parallel processing - scales with available cores
- Memory usage scales with `unique-tuples × relations × attributes`
- Partition count affects memory usage and parallelism
- Large domain sizes with high null percentages are most memory efficient

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{synthetic_join_datagen,
  title={Synthetic Join Data Generator},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## License

[Add your license information here]
