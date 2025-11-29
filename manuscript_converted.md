# RAG-based Architectures for Drug Side Effect Retrieval using compact LLMs

**Authors:**
Shad Nygren¹'*, Omer Erdogan¹'²'³'*, Pinar Avci¹, Andre Daniels¹, Reza Rassool¹, Afshin Beheshti¹'⁴'⁵, Diego Galeano¹'⁶'⁷⁺

**Affiliations:**
1. Kwaai, CA, USA
2. Department of Computer Engineering, Koc University, Istanbul, Turkey
3. KUIS AI Center, Istanbul, Turkey
4. Center for Space Biomedicine, McGowan Institute for Regenerative Medicine, Department of Surgery, University of Pittsburgh, Pittsburgh, Pennsylvania, USA
5. Broad Institute of MIT and Harvard, Cambridge, Massachusetts, USA
6. Department of Electronics and Mechatronics Engineering, Facultad de Ingeniería, Universidad Nacional de Asunción - FIUNA, Luque, Paraguay
7. Tesabio.ai Inc., DE, USA

\* These authors contributed equally.
⁺ Corresponding author: dgaleano@ing.una.py

---

## Abstract

Drug side effects are a major public health concern, yet off-the-shelf large language models (LLMs) are unreliable to accurately inform about reported drug side effects due to limited training data and domain gaps. We evaluate two open-book architectures that inject curated knowledge from the Side Effect Resource (SIDER 4.1) into LLM workflows: a text-based retrieval-augmented generation (RAG) pipeline and a graph-based variant (GraphRAG) over a Neo4j knowledge graph. On a balanced forward benchmark of 19,520 drug–side‑effect pairs, GraphRAG achieved 100% (Qwen‑2.5‑7B-Instruct) and 99.96% (Llama‑3.1‑8B-Instruct) accuracy; on reverse queries it returned exact drug sets with precision = recall = F1 = 100% at 0.09 s average latency (vs. text‑RAG Format B: F1 = 99.38%, 82.44 s). We also show that a lightweight LLM-based normalization step restores performance under common misspellings of drug names without modifying downstream logic. Taken together, these results indicate that integrating structured knowledge, especially graph representations, markedly improves LLM performance for drug side-effect retrieval, offering a practical path to interactive, evidence-grounded querying of catalogued drug side effect associations in larger language models.

---

## Introduction

Drug side effects represent a critical global public health challenge, significantly contributing to morbidity and mortality worldwide¹⁻⁴. The rapid pace of drug development often outstrips the capacity of healthcare professionals to stay abreast of new medication side effects, particularly outside their primary specialties⁵'⁶. This issue is further complicated when patients report potential adverse reactions, requiring physicians to assess causality during time-constrained appointments. Current tools, such as drug handbooks⁷, electronic medical records (EMRs)⁸, and Spontaneous Reporting Systems (FAERS)⁹'¹⁰, while valuable, require time-consuming search capabilities, underscoring the urgent need for more efficient and accessible resources for assessing drug side effects in clinical practice⁵'¹¹.

Large language models (LLMs)¹²⁻¹⁷ offer a promising avenue with their intuitive, conversational interfaces, illustrating the potential to streamline clinical workflows and enhance decision-making. These models support semantic search enabling the identification of drugs or diseases associated with specific symptoms¹⁸. Despite these advancements in natural language processing, the application of off-the-shelf LLMs in domain-specific tasks such as drug side effect identification has yielded mixed results¹⁹⁻²¹, frequently struggling with accuracy and reliability in specialized fields like pharmacovigilance¹⁴'²¹. Their limitations stem from knowledge constrained by black-box training data, a propensity for hallucinations, and a general lack of domain-specific expertise, which hinders the effectiveness of LLMs in handling nuanced medical data and generating contextually appropriate insights.

To overcome these significant challenges, we propose two open-book architectures designed to integrate domain knowledge about drug side effects into a Large Language Model (LLM): Retrieval Augmented Generation (RAG) and GraphRAG. Our first architecture utilizes RAG, which enhances LLMs by retrieving relevant information from an external Pinecone vector database—a HIPAA-compliant database —where drug side effect information is stored as feature vectors. The second architecture utilizes GraphRAG, which leverages a Neo4j graph database to store and efficiently bipartite drug side effect associations. Both frameworks incorporate custom split functions and filtering modules to optimize user prompts for accurate retrieval. Through evaluations on 19,520 drug–side-effect pairs, covering 976 marketed drugs and 3,851 MedDRA terms from the Side Effect Resource (SIDER) 4.1 database, we find that GraphRAG delivers very high accuracy for retrieving known associations under a binary yes/no formulation and under a single side effect query (reverse-query), substantially outperforming a lightweight closed-book LLM baseline and our text-based RAG variants. Within this evaluated setting, integrating structured knowledge—particularly a graph representation—markedly improves retrieval performance, providing a practical route to high-accuracy retrieval of catalogued side effects in LLM workflows.

Retrieval-augmented generation (RAG) architectures have been applied in biomedical and clinical contexts (e.g., BiomedRAG leveraging chunk-based retrieval³⁰) as well as KG-augmented LLMs (e.g., KG-RAG³¹ combining the SPOKE KG with LLM prompts). Throughout this work, we distinguish closed-book LLMs—models that answer without consulting external evidence (no retrieval; responses rely solely on parametric pre-trained knowledge)—from open-book LLMs, in which the model is provided retrieved evidence at inference time. Our open-book variants include text RAG (document chunks) and GraphRAG (Neo4j knowledge graph). While prior studies (e.g., KG-Rank, RAG²³², MKRAG³³) explore graph- or knowledge-infused retrieval, our contribution is not to propose RAG/Graph-RAG de novo. Rather, we adapt these concepts specifically to pharmacovigilance, use SIDER 4.1 as a benchmark, and provide a unified, head-to-head comparison of data representations (free text, pairwise, graph) under the same tasks.

---

## Results

### A Retrieval-Augmented Generation (RAG) framework for drug side effect retrieval

Our RAG system was designed for seamless retrieval of drug side effects, utilizing the Side Effect Resource (SIDER) 4.1 database²⁵ that contains drug side effect associations extracted from FDA public documents and drug package inserts. We filtered the database to include drugs with known Anatomical, Therapeutic, and Chemical (ATC) classification and side effects categorized as MedDRA Preferred Terms (PT). Filtering yielded a dataset of **141,209 associations**, linking **1,106 marketed drugs** to **4,073 unique side effect terms** (Fig. 1a). Due to the large number of these associations, utilizing the complete dataset for comprehensive evaluation would have been computationally prohibitive. For this assessment, we created a **balanced subset of 19,520 drug-side effect pairs**, as detailed in the Methods section.

To facilitate text-based retrieval, the raw SIDER dataset was processed into two distinct text formats (Fig. 1b):
- **"Text Format A"** provides a structured, comma-separated list of all known side effects for a given drug (e.g., "The drug metformin may be associated with the following side effects or adverse reactions: shock, peptic ulcer, contusion, …")
- **"Text Format B"** presents each drug-side effect pair on a new line, enhancing granularity (e.g., "The drug metformin may cause urticaria as an adverse effect, adverse reaction, or side effect.")

For the RAG pipeline (Fig. 1c, d), Text Format A was segmented into chunks using a custom algorithm that splits text at new lines. These chunks were then embedded into a **1,536-dimensional vector space** using the **OpenAI ada002 embedding model**, chosen for its capacity to support up to **8,192 tokens**, which is sufficient for even the longest text chunks in Format A (exceeding 10,000 characters). The resulting embeddings were indexed in a **Pinecone vector database**, enabling rapid similarity-based retrieval.

The RAG query workflow operates as follows: an end-user query (e.g., "Is urticaria an adverse effect of aspirin?") is first embedded using the OpenAI ada002 model and then compared against the **top five most similar entries** in the Pinecone database. Concurrently, an **LLM-based entity recognition module** extracts drug and side effect terms (e.g., "metformin" and "urticaria") from the query prompt. A subsequent **filtering module** checks if the identified drug-side effect pair from the query exists within the top five retrieved results. Based on this check, a modified prompt is generated: if a match is found, the prompt states that the side effect has previously been reported for the query drug; otherwise, it specifies that the drug is not known to be associated with the side effect.

**Prompt structure:**
```
"You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the information provided. Question:\n\n" + query + rag_results
```

The variable `rag_results` contains the result from RAG. For instance:
```
"No, the side effect " + side_effect_query + " is not listed as an adverse effect, adverse reaction or side effect of the drug " + drug_query
```

The modified prompt is then passed to a compact LLM model (**Llama‑3.1‑8B‑Instruct** or **Qwen‑2.5‑7B‑Instruct**) which generates a binary YES/NO response. This binary output was specifically chosen because our evaluation framed drug side effect retrieval as a binary classification problem.

### Graph-Based Retrieval Augmented Generation (GraphRAG) for Drug Side Effect Data

In our GraphRAG framework, drug-side effect associations are precisely modeled as a **bipartite graph-based representation**, leveraging the extensive SIDER 4.1 database previously described. Within this structure (Fig. 2a), drugs and side effects constitute distinct nodes, and their known relationships are encoded as directed edges, specifically labeled as **"may_cause_side_effect"**. This graph is implemented within a **Neo4j database**, a robust graph management system that facilitates efficient querying via Neo4j's query language, Cypher, enabling rapid traversal and retrieval of complex drug-side effect relationships.

The GraphRAG system is designed to process user queries, such as "Is headache an adverse effect of metformin?" (Fig. 2b). The workflow begins with an **entity recognition module** that extracts drug and side effect terms (e.g., "metformin" and "headache") from the submitted query. These extracted entities are then used to construct a precise Cypher query:

```cypher
cypher = f"""
    MATCH (s)-[r:may_cause_side_effect]->(t)
    WHERE s.name = 'metformin' AND t.name = 'headache'
    RETURN s, r, t
    """
```

This query is executed against the Neo4j database. It efficiently identifies the presence or absence of a direct edge between the specified drug and side effect nodes, returning matching associations or an empty result accordingly.

A **prompt engineering module** then processes the retrieved results to generate a context-specific input for the language model. If a match is found, the prompt is modified to state, "Metformin is known to be associated with headache as a side effect". Conversely, if no association is found, the prompt states, "Metformin is not known to be associated with headache as a side effect". This prompt modification strategy is identical to that employed in our RAG architecture, ensuring a consistent approach to informing the language model. This refined prompt is fed into a lightweight LLM (**Llama‑3.1‑8B‑Instruct** or **Qwen‑2.5‑7B‑Instruct**) model, which generates a binary YES/NO response. The system is served via **vLLM** with a lightweight REST API layer that orchestrates Neo4j graph lookups, supporting low-latency, scalable real-time queries.

This GraphRAG approach offers distinct advantages over traditional text-based retrieval methods for pharmacovigilance. By representing drug-side effect associations as a bipartite graph, it enables exact, relationship-driven queries that significantly reduce ambiguity and enhance retrieval accuracy. Its integration with Neo4j facilitates complex traversals, such as identifying groups of drugs associated with a query side effect, while the use of the LLM ensures future escalation to larger reasoning models that will provide user-friendly responses with more medical context.

### Performance evaluation for forward queries: from drug-side effect pair to binary answer

To quantify performance on drug–side-effect retrieval, we evaluate whether the model's final output to multiple single drug queries is correct against SIDER-derived ground truth. We constructed a **balanced evaluation set** by sampling **10 positives** (documented MedDRA Preferred Terms) and **10 negatives** (MedDRA terms not linked to that drug) for each drug with at least 10 known associations, yielding **19,520 pairs** across **976 drugs** and **3,851 side-effect terms**.

We compared four architectures:
1. **Closed-book LLM** (LLM-only, no retrieval from any database)
2. **Open-book LLM — RAG (Format A: drug→list of side effects)**
3. **Open-book LLM — RAG (Format B: drug-side effect pairs per sentence)**
4. **Open-book LLM — GraphRAG** (drug-side effect bipartite graph)

Using two compact models: **Qwen-2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**.

We report the accuracy, F1, precision, sensitivity, and specificity of the LLM's emitted YES/NO under each architecture.

#### Table 1: Comparative results (balanced SIDER subset; single-drug, binary queries) using Qwen2.5-7B

| Architecture type | Accuracy | F1 Score | Precision | Sensitivity | Specificity |
|-------------------|----------|----------|-----------|-------------|-------------|
| Closed-book LLM (no retrieval) | 62.90% | 0.494 | 0.776 | 0.363 | 0.895 |
| Open-book LLM — RAG (Format A: drug→list) | 86.67% | 0.858 | 0.919 | 0.805 | 0.928 |
| Open-book LLM — RAG (Format B: pairs) | 96.50% | 0.967 | 0.936 | 0.999 | 0.931 |
| Open-book LLM — GraphRAG | 100.00% | 1.000 | 1.000 | 1.000 | 1.000 |

#### Table 2: Comparative results (balanced SIDER subset; single-drug, binary queries) using Llama 3.1-8B-Instruct

| Architecture type | Accuracy | F1 Score | Precision | Sensitivity | Specificity |
|-------------------|----------|----------|-----------|-------------|-------------|
| Closed-book LLM (no retrieval) | 63.21% | 0.534 | 0.728 | 0.422 | 0.842 |
| Open-book LLM — RAG (Format A: drug→list) | 84.54% | 0.819 | 0.987 | 0.700 | 0.991 |
| Open-book LLM — RAG (Format B: pairs) | 95.86% | 0.960 | 0.924 | 0.999 | 0.918 |
| Open-book LLM — GraphRAG | 99.96% | 1.000 | 1.000 | 1.000 | 1.000 |

#### Table 3: LLM accuracy comparison for different architectures

| Architecture type | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct | Difference/winner |
|-------------------|---------------------|------------------------|-------------------|
| Closed-book LLM (no retrieval) | 62.90% | 63.21% | Llama +0.31% |
| Open-book LLM — RAG (Format A: drug→list) | 86.67% | 84.54% | Qwen +2.13% |
| Open-book LLM — RAG (Format B: pairs) | 96.50% | 95.86% | Qwen +0.64% |
| Open-book LLM — GraphRAG | 100.00% | 99.96% | Qwen +0.04% |

**Key findings:**
1. **Structured augmentation matters**: Both RAG formats and GraphRAG substantially outperform a pure LLM
2. **Data representation matters**: Pairwise (Format B) is consistently stronger than list-style (Format A)
3. **GraphRAG reaches the deterministic-lookup accuracy ceiling** (Qwen: 100.00%; Llama: 99.96%)

The tiny deviation from 100% reflects rare LLM mislabels (hallucinations) at the output step, not retrieval failures.

To further investigate whether the underperformance without data augmentation extends to significantly larger models, we also evaluated **ChatGPT 3.5** and **ChatGPT 4.0** on a subset of 51 randomly selected drugs in a closed-book setting. We observed a mean accuracy of approximately **55% for ChatGPT 3.5** and **63% for ChatGPT 4**. This demonstrates that even advanced, larger language models struggle to accurately identify drug side effects for marketed drugs without specialized augmentation.

### Performance evaluation for reverse queries: from side effect to drug set

In the previous section, we assessed the forward case: given a (drug, side-effect) pair, the system provides a binary YES/NO answer. We now turn to the **complementary, reverse case**: given a side-effect term, return the set of drugs known to be associated with it in SIDER. This shifts both user intent ("Which drugs cause side effect X?" vs. "Does drug D cause side effect X?") and evaluation: instead of a single label, the output is a set, so we measure precision, recall, and F1 over the returned drug lists, and we track latency.

To obtain these results, we constructed a **stratified benchmark** of side-effect terms spanning four tiers:
- **Rare** (5-19 drugs)
- **Small** (20-99 drugs)
- **Medium** (100-499 drugs)
- **Large** (500+ drugs)

We randomly sampled from each group for a total of **121 queries**. Ground truth was derived from the same SIDER-based database used in the forward task.

We compared three open-book variants:
1. RAG (Format A: drug to SE list)
2. RAG (Format B: pairs)
3. GraphRAG (Neo4j graph)

For each method, we executed the reverse query end-to-end and computed recall, precision, F1, and average latency.

#### Table 4: Reverse query performance (macro-averaged across tiers)

| Architecture | Recall | Precision | F1 | Avg Latency | Throughput |
|-------------|--------|-----------|----|-----------|-----------|
| Open-book LLM — GraphRAG (Neo4j) | 100.00% | 100.00% | 100.00% | 0.09 s | 11 q/s |
| Open-book LLM — RAG (Format B: pairs) | 98.88% | 99.93% | 99.38% | 82.44 s | 0.012 q/s |
| Open-book LLM — RAG (Format A: drug→list) | 7.97% | 80.91% | 11.84% | 23.42 s | 0.043 q/s |

**Key findings:**
- **GraphRAG** performs a single indexed Cypher expansion to enumerate all connected drugs and achieves exact coverage with near-instant latency
- **RAG (Format B)** approaches perfect F1 but slows dramatically as the number of matching drugs increases
- **RAG (Format A)** under-retrieves because list-style chunks are vulnerable to windowing/chunking limits

For prompts like "Which drugs cause [SE]?" we recommend **GraphRAG** as the default reverse-query backend.

### Misspelling Robustness and LLM-Assisted Normalization

In practical applications, user queries may contain misspelled drug names (e.g., "floxetine" for "fluoxetine"). Our current open-book pipelines are brittle under such misspellings, because the input string does not match a canonical name in our databases. To keep the core architectures unchanged and preserve their strengths, we introduce a **lightweight LLM-based entity-normalization layer** that operates between the user query and retrieval.

Concretely, upon receiving a user question, a compact LLM performs drug name correction/normalization to a canonical SIDER entry before any retrieval step. The normalized (drug, side-effect) is then passed to the same back-end used in the forward or reverse tasks.

To test this, we built a small database of **10 commonly misspelled drug names** previously reported in the literature. Our results indicate that our **Qwen 7B spell correction module achieves 80% accuracy** in this small set. We then ran our whole evaluation procedure for 9 out of the 10 drugs that we could map to our database.

#### Table 5: Comparison of different model/conditions to handle drug name misspelling

| Condition | RAG Format B (F1 score) | GraphRAG (F1 score) |
|-----------|------------------------|---------------------|
| Original architecture + no misspelled drugs | 0.9474 | 1.0000 |
| Original architecture + misspelled drugs | 0.0000 | 0.0000 |
| LLM-based Spelling normalization + misspelled drugs | 0.8333 | 0.8750 |

Our Qwen 7B spell correction achieved **~88% recovery** for both architectures, transforming a catastrophic 100% degradation into a manageable 12% degradation. This could be improved using larger LLMs.

---

## Discussion

We studied two ways of injecting curated knowledge from SIDER 4.1 into LLM workflows: a text‑based RAG pipeline and a graph‑based variant (GraphRAG). Using a large, balanced benchmark derived from SIDER, GraphRAG consistently reached the empirical ceiling for yes/no questions about catalogued drug–side‑effect pairs, and in reverse queries (side effect to drug set) it enumerated the exact set. Text RAG performed well but became slower as the number of matching drugs increased. Together, these results indicate that structured knowledge—particularly a graph representation—provides a reliable path to accurate, interactive retrieval when a curated reference resource is available.

Our aim was not to introduce RAG or Graph‑augmented LLMs de novo but to adapt them to pharmacovigilance and to compare representation formats under the same tasks and prompts. Framing the problem this way makes clear that performance is governed primarily by how knowledge is represented and retrieved, not by parametric scale alone. The graph schema, with explicit drug and side‑effect nodes linked by a single relation, reduces ambiguity and enables exact membership checks and set expansion with a simple query, while preserving a natural‑language interface for end users.

In the present design, correctness is determined upstream by retrieval and filtering; the LLM is retained to standardize the output and provide a concise, evidence-grounded rationale. This separation keeps the interface flexible for richer interactions (e.g., reverse queries now, class-level or multisource questions later) while allowing a "label-locking" guardrail when absolute fidelity is required. We used compact models to support exhaustive evaluation at low cost and latency; larger models can be swapped in when queries demand more reasoning or synthesis.

Our scope is intentionally narrow: we evaluate retrieval of previously catalogued associations within a closed set. The work does not address discovery of new adverse events, causal inference, or bias correction in spontaneous reporting. We also focus on single-drug forward queries and their reverse; class-based questions require additional ontology integration and dedicated benchmarks. Finally, exact-match stages are brittle to misspellings and brand–generic variation; a lightweight pre-retrieval normalization layer mitigates this without altering downstream logic.

We did not include a fine-tuned classifier baseline (e.g., BioBERT/RoBERTa trained on 141k SIDER pairs), prioritizing retrieval-based methods that leverage a complete knowledge store without memorization. While a supervised classifier could perform well on frequent pairs, it would likely trail RAG/GraphRAG on coverage and maintainability; we leave a systematic comparison to future work.

Looking ahead, the same interface can be extended along two axes:
1. **Query breadth**: Class-level prompts, ranked effect sets by frequency or severity, and multi-hop graph explanations
2. **Data breadth**: Integrating spontaneous reporting systems and literature via vector indices alongside the graph to surface emerging or rare signals

These extensions keep deterministic retrieval for fidelity and use the LLM to present evidence in language clinicians and patients can use.

---

## Methods

### Overview

We developed and evaluated two open-book architectures for drug–side-effect retrieval with large language models (LLMs):
1. **Text RAG** over SIDER-derived documents
2. **GraphRAG** over a Neo4j knowledge graph

We also include a **closed-book LLM** (no retrieval) baseline. Unless specified, all LLM inference was **zero-shot** (no fine-tuning) and served using **vLLM**.

**Terminology:**
- **"Closed-book LLM (no retrieval)"** = model answers from parametric memory only
- **"Open-book LLM"** = model receives retrieved evidence at inference time (RAG or GraphRAG)

### Data source and preparation

We used the **Side Effect Resource (SIDER) 4.1**²⁵, which compiles adverse events from randomized trials and post-marketing surveillance. We retained drugs with an **Anatomical, Therapeutic, and Chemical (ATC) code** and side effects mapped to **MedDRA Preferred Terms (PTs)**, yielding **141,209 associations** linking **1,106 marketed drugs** to **4,073 PTs**.

### Evaluation subset (forward YES/NO)

For forward binary evaluation, we constructed a balanced set as follows: for each drug with ≥10 PTs we sampled **10 positives** (documented PTs) and **10 negatives** (PTs not linked to that drug). This produced **19,520 pairs** spanning **976 drugs** and **3,851 PTs**. The full SIDER-derived graph (141,209 edges) was used for reverse-query ground truth.

### Knowledge representations

We derived three representations of SIDER for retrieval:
1. **Format A (drug→list)**: A per-drug narrative listing all PTs
2. **Format B (pairs)**: One sentence per (drug, PT) pair
3. **Graph**: A bipartite Neo4j graph with nodes = {Drug, SideEffect(PT)} and directed edges `:may_cause_side_effect`

### Entity recognition module

For the architectures, we extract drug and side effect names from the retrieved context using a **two-stage procedure**:
1. LLM-based extraction with **temperature of 0.1**
2. Regex-based parsing (to remove prefixes, parse comma-separated, parse lists, filter and remove duplicates if needed)

### Text RAG framework

**Indexing:** Format A documents were chunked at newlines and embedded (**OpenAI text-embedding-ada-002**, 1,536 dimensions). Format B pairs were individually embedded. All vectors were stored in Pinecone (serverless tier, cosine similarity).

**Retrieval:** User query is embedded; top-k vectors (k=5 for Format A, k=10 for Format B) are retrieved. Entity-recognition extracts (drug, SE); a filtering step checks if the pair appears in the retrieved chunks.

**LLM response:** The filtered result is passed to the LLM (Qwen-2.5-7B or Llama-3.1-8B) via vLLM, temperature=0.1, which emits a binary YES/NO label plus brief rationale.

### GraphRAG framework

**Graph construction:** Drugs and side-effects (PTs) are nodes; a directed edge `:may_cause_side_effect` connects each (drug, PT) pair from SIDER. The graph was loaded into Neo4j Aura Professional.

**Query:** Entity-recognition extracts (drug, SE) from the user prompt. A Cypher query checks for the edge:
```cypher
MATCH (d:Drug)-[:may_cause_side_effect]->(s:SideEffect)
WHERE d.name = $drug AND s.name = $se
RETURN d, s
```

**LLM response:** If a match is found, the prompt states "Drug X is known to be associated with side effect Y"; otherwise "Drug X is not known to be associated with side effect Y". This prompt is passed to the LLM (Qwen/Llama, temperature=0.1), which produces a YES/NO label.

### Evaluation metrics

For forward (binary) queries, we compute standard classification metrics:
- **Accuracy** = (TP + TN) / Total
- **Precision** = TP / (TP + FP)
- **Recall (Sensitivity)** = TP / (TP + FN)
- **Specificity** = TN / (TN + FP)
- **F1** = 2 × (Precision × Recall) / (Precision + Recall)

For reverse queries (set retrieval), we compute:
- **Recall** = |returned ∩ ground truth| / |ground truth|
- **Precision** = |returned ∩ ground truth| / |returned|
- **F1** = harmonic mean of precision and recall
- **Latency** = time from query submission to complete set materialization

### LLM inference

All models were served via **vLLM** (v0.3.1) with:
- **Temperature** = 0.1
- **Max tokens** = 512 (forward), 4096 (reverse)
- **top_p** = 0.9
- Models: **Qwen-2.5-7B-Instruct**, **Llama-3.1-8B-Instruct**

### Misspelling experiment

We compiled a list of 10 commonly misspelled drug names from the literature. For each misspelled query, we first passed it through an LLM-based normalization module (Qwen-7B, temperature=0.1) to correct the spelling to a canonical SIDER name. The corrected (drug, SE) pair was then evaluated through the standard forward pipeline. We report F1 score with and without the normalization layer.

---

## Data availability

All data generated or analyzed during this study are available in the Github link:
https://github.com/apicurius/drugRAG/tree/main/data/processed

## Code availability

The data and code used in our study are available here:
https://github.com/apicurius/drugRAG/tree/main

## Funding

This project was supported by Virtual Hipster Corporation.

## Acknowledgments

This project was supported by Virtual Hipster Corporation and Kwaai.

## Author Contribution

D.G. conceived the study and supervised the project. S.N, O.E. and D.G. designed the RAG frameworks. S.N. and O.E. implemented software architecture and ran experiments. P.A., A.D., and A.F. provided critical review and guidance on the medical and clinical aspects of the study. R.R. provided methods for analysis. D.G., P.A. and A.D. prepared the draft of the manuscript. All authors reviewed, edited, and approved the final manuscript.
