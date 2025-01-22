# README
<div style="overflow: auto;">
  <img src="./Hydrangea.png" alt="Alt text" width="300" style="float: right; margin-left: 10px;">
  <p>
  Hydrangea is a defect library for LLM-enabled software. Hydrangea has 4 main petals, each corresponding to one of the major components where defects often arise: LLM agent, vector database, software component, and system.
  </p>
</div>

Future updates for this repository will be available at [https://github.com/ShaoYuchen0112/Hydrangea](https://github.com/ShaoYuchen0112/Hydrangea).


## What is LLM-enabled software?

It is software that integrates LLMs (large language models) with RAG (retrieval-augmented generation ) support to realize intelligence features.

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
A suite of 100 non-trivial projects which tightly integrates LLMs and vector databases in their workflow.

We have uploaded `application.csv`, it contains:

   1. software project name
   2. GitHub link and commit ID
   3. classification
   4. used LLM and vector database

### Hydrangea Defect Library
The result of TABLE Ⅱ in our paper can be reproduced by this organized defect library. 

In the uploaded `defect.csv`, we have documented different cases for the same defect type, as defects can manifest in various ways. For each distinct case of the same defect, we have separated them with a blank line and labeled them as case 1, case 2, and so on, according to the specific circumstances.

It contains:

A collection of defects in these projects (involves 100 projects),containing
   1. the defect type and its detailed explanation
   2. the exact file and source-code line location of the defect
   3. the consequences of defect
   4. the defect-triggering tests




