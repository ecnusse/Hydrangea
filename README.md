# README
<div style="overflow: auto;">
  <img src="./Hydrangea.png" alt="Alt text" width="300" style="float: right; margin-left: 10px;">
</div>

**Yuchen Shao<sup>1,2</sup>**, **Yuheng Huang<sup>3</sup>**, **Jiawei Shen<sup>2</sup>**, **Lei Ma<sup>3,4</sup>**, **Ting Su<sup>2</sup>**, **Chengcheng Wan<sup>1,2</sup><sup>*</sup>**  
<sup>1</sup> Shanghai Innovation Institute  
<sup>2</sup> Software Engineering Institute, East China Normal University, Shanghai, China  
<sup>3</sup> Department of Computer Science, The University of Tokyo, Tokyo, Japan  
<sup>4</sup> University of Alberta, Edmonton, AB, Canada  

This is the artifact for our ICSE2025 paper "[Are LLMs Correctly Integrated into Software Systems?](https://arxiv.org/abs/2407.05138)". It is publicly archived at [Figshare](https://figshare.com/articles/dataset/Hydrangea/28262426). This repository is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://opendatacommons.org/licenses/pddl/1.0/ - see the [LICENSE](./LICENSE) file for details. Notably, a misintegration case studied in our paper has been officially assigned **CVE-2025-45150**. If you have any questions, please feel free to contact me via email(ycshao@stu.ecnu.edu.cn).

We are grateful for the contribution made by our anonymous collaborator. Additionally, **Mingyu Weng**, **Yiwen Sun**, and **Wenjing Liu** have developed the Command-line Interface (CLI) to further enhance its functionality. The `defect.csv` file has been updated accordingly. You can review the latest updates!

Hydrangea is a defect library for LLM-enabled software. Hydrangea has 4 main petals, each corresponding to one of the major components where defects often arise: LLM agent, vector database, software component, and system.

## What is LLM-enabled software?

It is software that integrates LLMs (large language models) with RAG (retrieval-augmented generation) support to realize intelligence features.

It contains four components:

1. **LLM agent** that manages LLM interfaces, constructs prompts, and invokes the
   LLM
2. **Vector database** that supports RAG algorithm and enhances the LLM agent
3. **Software component** that interacts with the first two components to perform certain tasks
4. **System** that manages resources and privileges to carry out the execution

## What's inside the artifact:

For enhanced availability and reusability, we offer an organized defect library utilized in our manual studies.

Below are details of what is included in each part:

### Application benchmark
A suite of 100 non-trivial projects that tightly integrates LLMs and vector databases in their workflow.

We have uploaded `application.csv`, which contains:

   1. software project name
   2. GitHub link and commit ID
   3. classification
   4. used LLM and vector database

### Hydrangea Defect Library
The result of TABLE Ⅱ in our paper can be reproduced by this organized defect library. 

In the uploaded `defect.csv`, we have documented different cases for the same defect type, as defects can manifest in various ways. For each distinct case of the same defect, we have separated them with a blank line and labeled them as case 1, case 2, and so on, according to the specific circumstances.

It contains:

A collection of defects in these projects (involves 100 projects), containing
   1. the defect type and its detailed explanation
   2. the exact file and source-code line location of the defect
   3. the consequences of defect
   4. the defect-triggering tests

The meaning of different columns in `defect.csv`:
   1. **APP**: the applications from GitHub.
   2. **commit url**: the relevant version of the application on GitHub.
   3. **types**: different defect types.
   4. **cases**: different examples for each defect type. Cells containing a "/" indicate that there is only one case for that defect type.
   5. **explanation**: details of the defect.
   6. **consequences**: the impacts of the defect. Here we use the abbreviations: ST refer to fail-stops, IC refer to incorrectness, SL refer to slower execution, UI refer to unfriendly user interface, TK refer to more tokens, and IS refer to insecure.
   7. **source-code locations**: The location of the code file where the defect occurs.
   8. **defect-triggering tests**: The software input that triggers the defect.

## Quick Start-How to use Hydranger?
<div style="overflow: auto;">
  <img src="./pic4tutorial.png" alt="Alt text" width="800" style="float: right; margin-left: 10px;">
</div>

Take **LocalAGI** as an example. It makes plans to guide users to achieve their goals. However, due to its infinite loop design with time intervals, it repeatedly refines a subset of the generated steps, without providing a final version that contains all the refinements. Making things worse, this loop could only be broken by terminating the entire application, significantly degrading user experience.

### Tutorial
   1. Open `application.csv` to find the corresponding GitHub link and commit ID for this application.
   2. Review `defect.csv` to get an overview of the defect and the associated defect-triggering tests.
   3. You can attempt to reproduce the issue.

## Command-line interface: Hydrangea command

### 🚀 Setup

1. Clone Defects4J:

   ```bash
   git clone https://github.com/ecnusse/Hydrangea.git
   ```
2. install dependencies
   ```bash
   pip install -e .
   ```

### 🎯 Command Overview

| Command | Description | Main Parameters |
|---------|-------------|-----------------|
| `apps` | 📱 List all applications, supports multi-dimensional filtering | `--classification`, `--llm`, `--vdb` |
| `bids` | 🐛 List all defect IDs, supports filtering by application | `--app` |
| `info` | 📊 Display metadata information of a specific defect | `app`, `bid` |
| `test` | 🧪 Display test information | `app`, `bid`, `--trigger` |


---

### 1. 📱 `apps` Command — List Applications

#### Basic Usage

```bash
# List all applications
hydrangea apps

# View detailed help information
hydrangea apps --help

# Filtering application based on llm
hydrangea apps --llm OpenAI

# Filtering application based on vector database
hydrangea apps --vdb chroma

# Filtering application based on language
hydrangea apps --language python

```



---

### 2. 🐛 `bids` Command — List Defect IDs

#### Basic Usage

```bash
# List all defect IDs
hydrangea bids
```

#### 🔍 Filter Defect IDs by Application

```bash
# List all defect IDs for a specific application
hydrangea bids --app LocalAGI
```

> 💡 **Tip**: Application names support fuzzy matching. All applications containing the specified keyword will be listed with their defect IDs.

---

### 3. 📊 `info` Command — View Detailed Defect Information

#### Basic Usage

```bash
# View detailed information for a specific defect
hydrangea info "camel" camel-ai-camel-privacy_violation-/
hydrangea info MEMGPT cpacker-MemGPT-exceeding_llm_content_limit-case1
```

#### 📋 Output Information Includes

- Defect Type
- Application Basic Information
- Defect Description and Impact


---

### 4. 🧪 `test` Command — Display Test Information
#### 📝 Parameter Description

| Parameter | Type | Description |
|------|------|------|
| `app` | Required | Application name |
| `bid` | Required | Defect ID |
| `--trigger` | Optional | Show detailed trigger test information |

#### 🚀 Usage Examples

```bash
# Show basic test information
hydrangea test camel camel-ai-camel-privacy_violation-/

# Show detailed trigger tests
hydrangea test camel camel-ai-camel-privacy_violation-/ --trigger
```

#### 📤 Output Examples

##### 🔸 Basic Mode Output

```bash
Test information for camel - camel-ai-camel-privacy_violation-/
Defect type: privacy violation
Case: /
Use --trigger to see detailed trigger tests
```

##### 🔸 Trigger Test Mode Output

```bash
trigger_tests:
- 1.Set up Camel: Ensure the project is correctly set up in your local environment.
2.Configure Roles: Set up a role-playing scenario where one agent has restricted privileges (e.g., "Guest User") and another has administrative privileges.
3.Define Security Boundaries: Establish clear system message constraints about access permissions.
4.Execute Test: Initiate conversation where the restricted role attempts to access privileged operations.
5.Observe Behavior: Monitor if the AI system properly enforces access controls or inadvertently reveals sensitive information.
```

---

### 5. 🔍 `analyze` Command — Analyze LLM Input/Output Format Issues

#### 📝 Description

This command automatically detects LLM input/output format issues in a specific software project. It clones the repository (if not already cloned), checks out to the specified commit, and analyzes the codebase for potential format problems in LLM interactions.

#### 🚀 Usage

```bash
# Analyze a specific application
hydrangea analyze <application_name>
```

#### 📊 What It Does

1. **Repository Management**: Automatically clones the repository from GitHub (if not already present locally) and checks out to the specific commit from `application.csv`(repos are stored in **./repos**)
2. **Code Analysis**: Scans the codebase to identify files that interact with LLMs
3. **Format Detection**: Uses Comfrey's `format_detector` to analyze:
   - Template discrepancies in LLM prompts and responses
   - Data segmentation issues
   - Context construction problems
4. **Results Reporting**: Provides a detailed summary of detected issues with severity levels (showing in the cli and creatinging a report in **./repos/analyze_report**)

#### 📤 Output Examples

```bash
Analyzing application: LocalAGI

✅ Analysis completed for VTSTech/localagi
🔗 Repository: https://github.com/VTSTech/localagi
📝 Commit ID: 85ecc246bd50783e9edaa4227493c0ac127912d8
📊 Classification: task management
🤖 LLM: OpenAI

📋 Analysis Summary:
  Total files analyzed: 15
  Files with format issues: 3

🔍 Detailed Format Issues:

File: localagi/agent.py
  📝 Prompt Issues:
    - Type: format_template_discrepancy
      Severity: Medium (0.50)
      Violations: 2
  📤 Completion Issues:
    - Type: format_data_segmentation
      Severity: Low (0.25)
      Violations: 1

File: localagi/utils.py
  📝 Prompt Issues:
    - Type: format_context_construction
      Severity: High (0.85)
      Violations: 3
```

#### 🔍 How It Works

The analysis leverages Comfrey's advanced format detection capabilities to identify three main types of issues:

1. **Template Discrepancy**: Detects mismatches between expected and actual formats in structured data (JSON, XML, YAML), positional templates, and code-fenced content
2. **Data Segmentation**: Analyzes word completeness, sentence integrity, boundary markers, and cross-segment coherence
3. **Context Construction**: Checks for relevance and coherence in context provided to LLMs

Each issue is assigned a severity score (0-1) to help prioritize fixes.

---
<div align="center">

**⭐ If this project is helpful to you, please give us a Star!**

Made with ❤️ by [Ungifted77](https://github.com/Ungifted77),[Evensunnn](https://github.com/Evensunnn),[SunsetB612](https://github.com/SunsetB612)

</div>
