# Comprehensive Guide to PULSE Agent Framework Evaluation

This guide provides a structured overview to conduct a thorough evaluation of the agents framework using the Jupyter notebooks in the `notebooks` directory. It details the available analyses, their purposes, and instructions for effective usage.

---

## Directory Structure and Purpose

The `notebooks` directory contains modular notebooks for:
- **Exploratory Data Analysis (EDA)**
- **Agent-Specific Evaluation**
- **Overarching and Operational Metrics**

Each notebook is designed for a specific stage of the evaluation pipeline, from initial data inspection to advanced agent reasoning analysis.

---

## 1. Exploratory Data Analysis

**Notebook:** `exploratory_data_analysis.ipynb`

**Purpose:**
- Explore harmonized datasets prior to modeling.
- Assess data structure, missingness, feature distributions, and cohort demographics.

**Key Analyses:**
- Dataset loading and merging
- Venn diagrams for patient/task overlap
- Missing data heatmaps
- Feature distribution plots
- Temporal coverage histograms
- Summary statistics tables

**Usage:**
- Execute all cells sequentially.
- Outputs are saved to `notebook_output/exploratory_data_analysis/`.

---

## 2. Output Data Postprocessing

**Notebook:** `pulse_results_data_postprocessing.ipynb`

**Purpose:**
- Postprocess metadata and calculate performance/operational metrics for all models and agents.

**Key Analyses:**
- Sample alignment and demographics mapping
- Calculation of AUROC, AUPRC, and other metrics
- Subgroup analyses (sex, age, BMI)
- Aggregated operational metrics (tokenization/inference time, token counts)
- Publication-ready metrics tables

**Usage:**
- update `outputfolder_path_list` and `get_llm_prompting_id_paths` to correctly point towards the output folders that contain `..._metadata.csv` files
- Run the notebook to generate CSV tables and summary statistics.
- Outputs are saved to `notebook_output/postprocessed_data/` and `notebook_output/metrics_tables/`.

---

## 3. Overall Agents Framework Evaluation

**Notebook:** `pulse_agents_overall_eval.ipynb`

**Purpose:**
- Provides a comprehensive, non-agent-specific evaluation of the entire agents framework, including all models and prompting strategies.
- Focuses on comparative benchmarking, fairness, calibration, operational efficiency, and cost analysis across all agents and baselines.

**Key Analyses:**
- **Performance Metrics:** Bar charts and grids comparing AUROC, AUPRC, and other metrics across tasks, datasets, and models.
- **Subgroup Fairness:** Visualizations and summaries of fairness deviations by sex, age, BMI, and other subgroups.
- **Radar Charts:** Multi-metric and multi-task radar plots for holistic model comparison.
- **ROC & PRC Curves:** 3x3 grid visualizations for all task-dataset combinations.
- **Calibration Analysis:** Reliability diagrams, calibration error plots, and actionable calibration summaries.
- **Prediction Distribution:** Histograms and density plots for predicted probabilities across models and strategies.
- **Operational Metrics:** Heatmaps and scatter plots for tokenization/inference time, throughput, and efficiency.
- **Cost Analysis:** Token usage, cost breakdowns, and cost-performance tradeoff visualizations.
- **Summary Tables:** Actionable insights and summary tables for reporting and publication.

**Usage:**
- Run the notebook sequentially to generate all comparative analyses and visualizations.
- Outputs are saved to `notebook_output/pulse_agents_overall/` and its subdirectories (e.g., `radar_charts`, `cost_analysis`, `operational_performance`).
- Use this notebook for high-level benchmarking, fairness, and operational/cost analysis across all agents and baselines.
- Review the generated figures and tables for publication, reporting, or further

---

## 4. Agent-Specific Evaluation

Agent evaluation notebooks provide in-depth analysis of reasoning, performance, and workflow for each agent type.

### Example: Hybrid Reasoning Agent

**Notebook:** `pulse_agent_hra_eval.ipynb`

**Purpose:**
- Analyze the Hybrid Reasoning Agent (HybReAgent) in detail.

**Key Analyses:**
- Metadata overview
- Step-by-step reasoning and prediction evolution
- Agreement analysis (ML vs clinical assessment)
- Investigation triggers and synthesis quality
- Feature importance and overlap
- Qualitative assessment and confusion flow
- Preparation for clinician validation

**Usage:**
- Run all cells in order.
- Outputs are saved to `notebook_output/pulse_agents_hybreagent/`.

### Other Agents

- Notebooks such as `pulse_agents_cra_eval.ipynb`, `pulse_agents_clinflow_eval.ipynb`, etc., follow similar structures for their respective agents.
- Outputs are saved to corresponding subdirectories under `notebook_output/`.

---

## 5. General Workflow

1. **Begin with EDA:**
   - Run `exploratory_data_analysis.ipynb` to inspect data quality and cohort structure.

4. **Benchmarking:**
   - Use `pulse_results_data_postprocessing.ipynb` for postprocessing of output data, which is used as input for the remaining notebooks.

3. **Agent Evaluation:**
   - Select and run the relevant agent notebook for detailed analysis.

---

## 6. Output Organization

| Analysis Type                         | Notebook Name                              | Output Location                                 |
|----------------------------------------|--------------------------------------------|-------------------------------------------------|
| Data Exploration & Statistics          | `exploratory_data_analysis.ipynb`          | `notebook_output/exploratory_data_analysis/`    |
| Output Postprocessing  | `pulse_results_data_postprocessing.ipynb`  | `notebook_output/postprocessed_data/`, `notebook_output/metrics_tables/` |
| Overall Agents Framework Evaluation    | `pulse_agents_overall_eval.ipynb`          | `notebook_output/pulse_agents_overall/`         |
| HybReAgent Evaluation      | `pulse_agent_hra_eval.ipynb`               | `notebook_output/pulse_agents_hybreagent/`      |
| ClinFlowAgent Evaluation               | `pulse_agents_cwa_eval.ipynb`          | `notebook_output/pulse_agents_clinflowagent/`         |
| ColAgent Evaluation    | `pulse_agents_cra_eval.ipynb`              | `notebook_output/pulse_agents_cra/`             |
| Model Architecture Visualization       | `convDL_network_visualizations.ipynb`      | `notebook_output/convDL_architectures/`         |

---

