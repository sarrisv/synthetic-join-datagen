# Synthetic Join Data & Plan Generator

A configurable, parallelized tool for generating synthetic relational datasets, query execution plans, and join selectivity analysis. Designed for database testing, research, and benchmarking.

## Features

-   **Parallel Data Generation**: Uses Dask to efficiently generate large datasets across multiple cores.
-   **Configurable Data Distributions**: Create realistic data with `uniform`, `zipf`, and `normal` distributions.
-   **Rich Data Parameters**: Control the number of relations, attributes, tuple counts, duplication, domain size, and null percentages.
-   **Versatile Join Patterns**: Generate plans using `star`, `chain`, `cyclic`, and `random` patterns.
-   **Multiple Plan Granularities**: Output plans in both `table-at-a-time` (binary joins) and `attribute-at-a-time` (multi-way joins) formats.
-   **On-the-Fly Analysis**: Automatically compute detailed join selectivity analysis as plans are generated.
-   **Flexible Output Formats**:
    -   Data: `csv`, `json`, `parquet`
    -   Plans: `txt`, `json`
-   **Reproducibility**: Use a global seed for fully reproducible generation runs.
-   **Configuration Flexibility**: Manage runs via rich command-line arguments or a simple TOML configuration file.
-   **Extensible Architecture**: Easily add new data distributions or join patterns via a registry system.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd synthetic-join-datagen

# Install dependencies using uv
uv sync
```

## Quick Start

### 1. Using Command-Line Arguments

This command generates a small dataset with 3 relations, runs on-the-fly analysis, and enables verbose logging.

```bash
python src/main.py \
    --relations 3 \
    --unique-tuples 5000 \
    --distribution zipf \
    --dist-skew 1.5 \
    --plans 5 \
    --join-pattern star chain \
    --plan-granularity table attribute \
    --run-on-the-fly-analysis \
    -v
```

### 2. Using a Configuration File

For complex or reproducible runs, use a `TOML` configuration file.

1.  Create a file named `config.toml`:

    ```toml
    # config.toml

    [data]
    relations = 5
    attributes = 4
    unique_tuples = 10000
    distribution = "zipf"
    dist_skew = 2.0
    domain_size = 5000
    null_percentage = 0.1
    data_output_format = "parquet"

    [plan]
    plans = 10
    join_pattern = ["star", "random"]
    plan_granularity = ["table", "attribute"]
    generate_dot_visualization = true

    [analysis]
    run_on_the_fly_analysis = true

    [system]
    seed = 12345
    verbose = 1
    ```

2.  Run the generator with the config file:

    ```bash
    python src/main.py --config-file config.toml
    ```

    *Note: When using `--config-file`, only `--verbose` and `--log-file` can be used as additional command-line flags.*

## Command-Line Reference

The tool provides a comprehensive set of command-line arguments. For a full, up-to-date list of all options and their defaults, run:

```bash
python src/main.py --help
```

## Output Structure

All generated files are placed in the `base_output_dir` (defaults to `generated_output/`).

```
generated_output/
├── data/
│   ├── R0.csv
│   ├── R1.csv
│   └── ...
├── plans/
│   ├── plan_0_star_table.txt
│   ├── plan_0_star_table.json
│   ├── plan_1_random_attribute.txt
│   ├── plan_1_random_attribute_viz.dot
│   └── ...
└── analysis/
    ├── analysis_plan_0_star_table.json
    ├── analysis_plan_1_random_attribute.json
    ├── ...
    ├── selectivity_analysis_report.txt      # Aggregated human-readable report
    └── selectivity_analysis_data.json       # Aggregated machine-readable data
```

-   **`data/`**: Contains the generated relation data files (e.g., `R0.csv`).
-   **`plans/`**: Contains the derived plan files (`.txt`, `.json`) and optional DOT visualizations (`.dot`).
-   **`analysis/`**:
    -   Individual `analysis_plan_*.json` files are created for each plan during the run.
    -   At the end of a successful run, these are aggregated into `selectivity_analysis_report.txt` and `selectivity_analysis_data.json`.

## Architecture

The project is organized into modular components:

-   `main.py`: Main entry point, orchestrates the Dask cluster and generation workflow.
-   `cli.py`: Defines the command-line interface using `argparse` and handles `TOML` configuration loading.
-   `data_generation.py`: Contains the logic for generating relational data in parallel with Dask.
-   `plan_generation.py`: Logic for generating base join patterns and deriving plans of different granularities.
-   `analysis_module.py`: Core logic for computing join selectivities and aggregating results.
-   `plan_structures.py`: Dataclasses defining the structure of Plans and Stages.
-   `registries.py`: Central registry for `DataDistribution`, `JoinPattern`, and `Formatter` implementations, making the tool easily extensible.

## Extending the Tool

The use of registries makes it simple to add new functionality without modifying the core logic.

### Adding a New Data Distribution

1.  Create a new class that inherits from `DataDistribution`.
2.  Implement the `generate_values` method.
3.  Register your new class in `registries.py`.

```python
# In registries.py

class MyCustomDistribution(DataDistribution):
    def generate_values(self, rng, values, domain_size, dist_args=None):
        # Your custom logic here
        ...

DISTRIBUTIONS["my_custom"] = MyCustomDistribution
```

### Adding a New Join Pattern

1.  Create a new class that inherits from `JoinPattern`.
2.  Implement the `generate_joins` method.
3.  Register your new class in `registries.py`.

```python
# In registries.py

class MyCustomPattern(JoinPattern):
    def generate_joins(self, rng, relations, num_attrs):
        # Your custom logic here
        ...

JOIN_PATTERNS["my_custom"] = MyCustomPattern
```

## Citation

If you use this tool in your research, please consider citing it.

```bibtex
@software{synthetic_join_datagen_2025,
  author = {Vasilis Sarris},
  title = {{Synthetic Join Data & Plan Generator}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {\url{https://github.com/sarrisv/synthetic-join-datagen}}
}
```

## License

[MIT License]
