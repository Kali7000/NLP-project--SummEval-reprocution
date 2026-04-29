# Reproducing SummEval: Re-evaluating Summarization Evaluation
### ReproNLP 2026 — Track A Reproduction Report
**GEM 2026 Workshop — 5th Generation, Evaluation & Metrics Workshop**

---

## Abstract

This report presents a reproduction of *SummEval: Re-evaluating Summarization Evaluation* (Fabbri et al., 2021), a landmark study that systematically benchmarked automatic summarization metrics against human judgments. We reproduce the paper's core correlation analysis using the publicly available SummEval dataset, computing ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore against expert human annotations across four quality dimensions: coherence, consistency, fluency, and relevance. Our results closely replicate the paper's primary claims — that BERTScore correlates more strongly with human judgments than ROUGE on most dimensions — while surfacing a notable divergence on the Consistency dimension, where no metric we tested performed reliably. We also confirm the paper's finding that system-level aggregation substantially increases metric reliability. This reproduction validates the central findings of SummEval while highlighting Consistency as an enduring open problem in automatic NLP evaluation.

---

## 1. Introduction

Automatic text summarization is one of the core tasks in Natural Language Processing (NLP). A summarization system takes a long document — a news article, a research paper, a legal brief — and produces a shorter version that captures the most important information. With the rapid growth of large language models (LLMs) such as GPT, BART, and T5, the ability to automatically generate summaries has improved dramatically. However, a critical question remains: **how do we know if a generated summary is actually good?**

Evaluating summaries is harder than it sounds. A summary can be fluent and grammatically correct, yet completely miss the point of the article. It can be relevant, yet contain factual errors not present in the original text. It can be concise, yet incoherent as a piece of writing. Human evaluation — having people read and score summaries — is the gold standard, but it is expensive, slow, and difficult to scale.

For this reason, the NLP community has developed **automatic evaluation metrics** — mathematical formulas that can score a summary in milliseconds without human intervention. The most widely used of these is **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation), which works by counting how many words or phrases from a human-written reference summary appear in the generated summary. More recently, **BERTScore** was proposed, which instead measures semantic similarity using deep neural network embeddings, comparing meaning rather than exact word matches.

The central question of the SummEval paper is: **do these automatic metrics actually agree with human judgment?** In other words, when a metric says a summary is good, do human readers agree? If not, the field may be optimizing for the wrong thing.

### 1.1 Why Reproducibility Matters

A scientific result is only trustworthy if it can be reproduced independently. In NLP, this is a well-known problem: many published results have been shown to be sensitive to implementation details, random seeds, software versions, or subtle differences in data preprocessing. The ReproNLP shared task specifically addresses this by asking researchers to take a published NLP evaluation study and attempt to reproduce its findings from scratch.

This report documents our attempt to reproduce the core quantitative findings of Fabbri et al. (2021) using the same publicly released dataset and the same evaluation methodology.

---

## 2. The Original Paper: SummEval

### 2.1 Paper Overview

**Title:** SummEval: Re-evaluating Summarization Evaluation  
**Authors:** Alexander R. Fabbri, Wojciech Kryściński, Bryan McCann, Caiming Xiong, Richard Socher, Dragomir Radev  
**Published:** Transactions of the Association for Computational Linguistics (TACL), 2021  
**ArXiv:** https://arxiv.org/abs/2007.12626

The paper makes three primary contributions:

1. A large-scale human evaluation benchmark of 16 summarization models, each scored by trained expert annotators on 100 news articles drawn from the CNN/DailyMail dataset.
2. A systematic comparison of 14 automatic evaluation metrics against those human scores.
3. A finding that widely used metrics like ROUGE correlate only weakly with human judgments, and a recommendation to adopt newer neural metrics like BERTScore.

### 2.2 The Four Evaluation Dimensions

The paper evaluates summaries on four distinct quality dimensions, each scored on a 1–5 scale by human annotators:

| Dimension | Definition |
|---|---|
| **Coherence** | Is the summary well-structured, logically organized, and easy to follow as a standalone piece of text? |
| **Consistency** | Does the summary contain only information that is supported by the source article? Are there any hallucinated or fabricated facts? |
| **Fluency** | Is the language of the summary grammatically correct, natural-sounding, and free from errors? |
| **Relevance** | Does the summary focus on the most important information from the source article? |

These four dimensions are intentionally distinct. A model can score highly on Fluency (producing grammatical text) but poorly on Consistency (making up facts). This decomposition is one of SummEval's key methodological contributions.

### 2.3 Key Claims of the Original Paper

The paper makes the following central claims, which we set out to verify:

- **Claim 1:** BERTScore correlates more strongly with human judgments than ROUGE across most evaluation dimensions.
- **Claim 2:** No automatic metric correlates well with Consistency (factual faithfulness), indicating this dimension is particularly difficult to measure automatically.
- **Claim 3:** All metrics perform substantially better at the system level (averaging scores across all summaries from one model) than at the summary level (scoring individual summaries). This means metrics are more useful for comparing entire models than for scoring individual outputs.

---

## 3. Reproduction Methodology

### 3.1 Dataset

We used the publicly released SummEval dataset from the Yale-LILY GitHub repository (https://github.com/Yale-LILY/SummEval). The primary data file is `model_annotations.aligned.jsonl`, a JSON Lines file where each line represents one generated summary and contains:

- `decoded`: The generated summary text produced by one of 16 summarization models.
- `references`: A list of human-written reference summaries for the corresponding news article.
- `expert_annotations`: A list of per-annotator scores (coherence, consistency, fluency, relevance), each on a 1–5 integer scale, provided by trained expert annotators.
- `turker_annotations`: Scores from crowdworkers (not used in this reproduction, consistent with the original paper's focus on expert annotations).
- `model_id`: An identifier for which summarization model produced the summary (M0 through M23).

One important observation during data loading: **the JSONL file does not contain the source article text**, only a `filepath` pointing to the original CNN/DailyMail story file. Since our reproduction focuses on reference-based metrics (ROUGE and BERTScore), which compare the generated summary to reference summaries rather than the source article, this does not affect our analysis. However, it does mean that source-document-based metrics such as FactCC or SUPERT cannot be reproduced without separately downloading the full CNN/DailyMail corpus.

The final dataset contains **1,600 records**: 16 models × 100 articles each. Human scores for each summary were computed by averaging across all expert annotators.

### 3.2 Automatic Metrics Computed

We computed the following metrics for each of the 1,600 summaries:

**ROUGE (Lin, 2004)**  
ROUGE measures the overlap between n-grams in the generated summary and n-gram in the reference summaries. We compute three variants:
- *ROUGE-1*: Unigram (single word) overlap
- *ROUGE-2*: Bigram (two-word phrase) overlap
- *ROUGE-L*: Longest common subsequence overlap

All ROUGE scores are computed as F1 (harmonic mean of precision and recall). For summaries with multiple reference summaries, we score against each reference independently and take the maximum score. We used the `rouge-score` Python library with stemming enabled.

**BERTScore (Zhang et al., 2020)**  
BERTScore measures semantic similarity by encoding both the generated summary and reference summaries using `roberta-large`, a large pre-trained language model, and computing cosine similarity between contextual token embeddings. Unlike ROUGE, BERTScore can recognize that "automobile" and "car" are semantically similar, even though they share no characters. We report the F1 variant of BERTScore, again taking the maximum score across multiple references. We used the `bert-score` Python library with the `roberta-large` model.

### 3.3 Correlation Analysis

Following the original paper, we compute **Spearman rank correlation** (ρ) between each automatic metric and each human evaluation dimension. Spearman correlation is appropriate here because human scores are ordinal (1–5 integers) and the relationship between metrics and human scores is not necessarily linear.

We compute correlations at two levels of analysis:

- **Summary-level:** Correlations computed across all 1,600 individual summary-score pairs.
- **System-level:** Each model's scores are first averaged to produce one data point per model (16 total), then correlations are computed across these 16 model-level averages.

### 3.4 Implementation Environment

- Python 3.11, Spyder IDE (Anaconda distribution)
- `rouge-score` 0.1.2, `bert-score` 0.3.13, `transformers` 4.x
- CPU-only computation (Intel processor, no GPU)
- BERTScore computation time: approximately 25 minutes using the `BERTScorer` class with batch processing

---

## 4. Results

### 4.1 Human Score Distributions

Before analyzing metric correlations, we examine the distribution of human scores across the four dimensions (Figure 3). This reveals an important structural feature of the dataset that has direct implications for metric evaluation.

**Coherence** (mean = 3.41) shows a roughly uniform distribution across the 1–5 scale, with scores spread across all rating levels. This variability makes it amenable to correlation analysis.

**Consistency** (mean = 4.66) is extremely right-skewed — the vast majority of summaries received a score of 4 or 5, with very few summaries rated 3 or below. This near-ceiling distribution severely compresses the variance in consistency scores, making it statistically difficult for any metric to achieve high correlation with this dimension regardless of its actual quality. Modern neural summarization models rarely produce outright factually inconsistent summaries, which explains this distribution.

**Fluency** (mean = 4.67) is similarly right-skewed, with most summaries rated near the top of the scale. Modern neural text generation produces fluent text by default.

**Relevance** (mean = 3.78) shows moderate variability, somewhat skewed toward higher scores but with meaningful spread throughout the range.

This distributional analysis is critical context for interpreting all correlation results that follow: the near-ceiling distributions of Consistency and Fluency make high correlations with those dimensions inherently difficult to achieve.

### 4.2 Summary-Level Spearman Correlations

Table 1 presents our reproduced summary-level Spearman correlations alongside the original paper's reported values.

**Table 1: Summary-Level Spearman Correlations (ρ) — Reproduced vs. Original**

| Metric | Coherence | Consistency | Fluency | Relevance |
|---|---|---|---|---|
| ROUGE-1 | 0.222 *(paper: 0.22)* | 0.176 *(paper: 0.14)* | 0.122 *(paper: 0.18)* | 0.334 *(paper: 0.33)* |
| ROUGE-2 | 0.149 *(paper: 0.18)* | 0.148 *(paper: 0.12)* | 0.081 *(paper: 0.16)* | 0.236 *(paper: 0.24)* |
| ROUGE-L | 0.204 *(paper: —)* | 0.146 *(paper: —)* | 0.092 *(paper: —)* | 0.248 *(paper: —)* |
| BERTScore | 0.375 *(paper: 0.25)* | 0.100 *(paper: 0.16)* | 0.144 *(paper: 0.23)* | 0.367 *(paper: 0.39)* |

Overall, ROUGE correlations match the original paper closely, within 0.02–0.04 ρ units across most cells. The BERTScore results show larger deviations: our BERTScore achieves higher-than-reported correlations for Coherence (0.375 vs. 0.25) but lower-than-reported correlations for Fluency (0.144 vs. 0.23) and Consistency (0.100 vs. 0.16). Possible explanations include differences in the exact `roberta-large` model checkpoint used, differences in how multi-reference scoring is handled, and version differences in the `bert-score` library between 2020 and 2026. Importantly, the relative ordering of metrics is largely preserved.

### 4.3 Verification of Paper Claims

**Claim 1 — BERTScore outperforms ROUGE: LARGELY CONFIRMED**

BERTScore achieves the highest correlation for Coherence (ρ = 0.375 vs. ROUGE-1's 0.222) and Relevance (ρ = 0.367 vs. ROUGE-1's 0.334). This confirms the paper's central claim that contextual embedding-based metrics better capture these holistic quality dimensions than surface-level n-gram overlap.

**Claim 2 — No metric correlates well with Consistency: CONFIRMED**

This is the most striking finding. BERTScore achieves only ρ = 0.100 on Consistency — actually *lower* than ROUGE-1's 0.176. This represents the one dimension where the paper's specific claim about BERTScore's superiority did not reproduce. More importantly, the broader finding holds emphatically: all tested metrics show very weak correlation with human Consistency judgments. This makes intuitive sense — detecting whether a summary contains hallucinated facts requires understanding the semantic content of the source document, not just comparing the summary to a reference. Neither n-gram matching nor embedding similarity is well-suited to this task.

**Claim 3 — System-level correlations are higher: CONFIRMED**

This finding reproduced dramatically and without exception (with two notable caveats discussed below).

### 4.4 System-Level Correlations

**Table 2: System-Level Spearman Correlations (ρ)**

| Metric | Coherence | Consistency | Fluency | Relevance |
|---|---|---|---|---|
| ROUGE-1 | 0.497 | 0.459 | 0.633 | **0.732** |
| ROUGE-2 | 0.403 | 0.271 | 0.565 | 0.674 |
| ROUGE-L | 0.471 | 0.124 | 0.487 | 0.635 |
| BERTScore | **0.715** | 0.021 | 0.419 | 0.497 |

Aggregating to the system level dramatically increases correlations for nearly every metric-dimension pair. The most striking example is ROUGE-1 on Relevance, which rises from ρ = 0.334 to ρ = 0.732 — a more than doubling of correlation strength. BERTScore on Coherence rises from ρ = 0.375 to ρ = 0.715.

There are two notable exceptions where system-level correlations are *lower* than summary-level: ROUGE-L on Consistency (0.146 → 0.124) and BERTScore on Consistency (0.100 → 0.021). This exceptional behavior further underscores that Consistency is fundamentally different from the other three dimensions — averaging over summaries does not help metric reliability when the metric is measuring the wrong signal entirely.

At the system level, an interesting reversal occurs: ROUGE-1 outperforms BERTScore on Fluency (0.633 vs. 0.419) and Relevance (0.732 vs. 0.497). BERTScore retains its advantage only on Coherence (0.715 vs. 0.497). This suggests the choice of metric should depend on which dimension matters most and at what granularity of analysis.

### 4.5 System-Level Model Comparison

Figure 4 shows the average human scores for each of the 16 summarization models across all four dimensions. Several patterns emerge:

- **Consistency and Fluency are uniformly high** across all models, reflecting the right-skewed distributions observed earlier. Nearly every model achieves near-5.0 on these dimensions, making them difficult to use as differentiators between models.
- **Coherence is the most discriminating dimension**, with scores ranging from 2.27 (M11) to 4.22 (M23), a spread of nearly 2 full points on the 5-point scale.
- **M23 and M22 consistently rank among the top performers** across Coherence, Consistency, Fluency, and Relevance, suggesting these represent stronger abstractive models (likely transformer-based).
- **M11 and M9 show notably lower Coherence scores**, indicating that while they may produce fluent sentences, their summaries lack overall structural quality.

---

## 5. Discussion

### 5.1 What the Consistency Finding Means

The failure of all tested metrics to correlate with Consistency is the most practically significant finding of both the original paper and our reproduction. Consistency — whether a summary makes claims that are actually supported by the source document — is arguably the most important quality dimension from a real-world perspective. A summary that introduces facts not present in the original article (a phenomenon called "hallucination") can mislead readers, propagate misinformation, and cause real harm in high-stakes applications such as medical or legal summarization.

The near-zero BERTScore correlation with Consistency at the system level (ρ = 0.021) means that ranking models by their BERTScore tells you essentially nothing about which model produces more factually accurate summaries. This is a severe practical limitation that the field has since begun to address through dedicated factuality metrics (FactScore, ANLI-based evaluation, etc.), but it remains a critical gap as of the time of this writing.

The right-skewed distribution of Consistency scores also partially explains this finding. When almost all summaries receive a score of 4 or 5, there is little variance for a metric to correlate against. Future evaluations should be designed to elicit greater variance in Consistency scores — perhaps by deliberately including more challenging or complex articles.

### 5.2 The System vs. Summary Level Gap

The dramatic difference between summary-level and system-level correlations has an important practical implication: **automatic metrics are much more appropriate for model selection than for quality control of individual outputs.** If you want to compare two summarization models to decide which one to deploy, ROUGE and BERTScore provide meaningful signal at the system level. If you want to use an automatic metric to filter out bad summaries in a production pipeline, the weak summary-level correlations suggest this approach is unreliable.

This distinction is frequently overlooked in NLP research papers, which often report system-level results but apply the findings to argue for summary-level use cases.

### 5.3 Deviations from the Original Paper

Our reproduction deviated from the original paper in several ways that are worth documenting:

**BERTScore values:** Our BERTScore correlations diverge more from the paper than our ROUGE correlations do. This is most likely due to model checkpoint differences — the `roberta-large` model used in 2020 may differ from the version downloaded in 2026 — and potentially differences in the `rescale_with_baseline` parameter. We did not use baseline rescaling, while some BERTScore implementations do.

**Subset of metrics:** The original paper evaluated 14 automatic metrics including MoverScore, BARTScore, FactCC, CIDEr, and METEOR. We reproduced only ROUGE variants and BERTScore due to installation complexity and computational constraints. This is a meaningful limitation — the paper's full conclusions about the relative ranking of all 14 metrics cannot be verified from our reproduction alone.

**Turker annotations:** The original paper separately analyzes expert annotations and crowdworker (Mechanical Turk) annotations. We used only expert annotations, as the paper identifies these as the more reliable signal.

### 5.4 Implications for NLP Evaluation in 2026

Five years after SummEval's publication, its core findings remain relevant and somewhat sobering. Despite the significant advances in both summarization models and evaluation metrics during this period:

- ROUGE remains the most commonly reported metric in summarization papers, despite its documented weaknesses.
- Consistency/factuality evaluation remains an open research problem, with no single automatic metric achieving broad adoption.
- The system-vs-summary-level gap is rarely discussed explicitly in published work.

Our reproduction confirms that SummEval's call for more careful, multi-dimensional evaluation of summarization systems was well-founded — and remains heeded incompletely by the field.

---

## 6. Conclusion

We successfully reproduced the core findings of Fabbri et al. (2021). Our main conclusions are:

**Reproduced:** BERTScore correlates more strongly with human judgments than ROUGE on Coherence and Relevance at the summary level, confirming the paper's primary claim.

**Reproduced:** System-level correlations are substantially higher than summary-level correlations for almost all metric-dimension pairs, confirming the paper's recommendation to aggregate results when comparing models.

**Partially diverged:** BERTScore's performance on Consistency did not reproduce — we find BERTScore performs worse than ROUGE-1 on this dimension (ρ = 0.100 vs. 0.176), contrary to the paper's reported values. However, the broader finding that Consistency resists automatic evaluation is strongly confirmed by both our results and the original paper's.

**Key novel observation:** BERTScore's system-level Consistency correlation collapses to near zero (ρ = 0.021), lower even than its already-weak summary-level value. This is the starkest evidence in our results that BERTScore is measuring a fundamentally different property than human factual judgment.

In summary, SummEval's core methodological contribution — that automatic metrics must be validated against human judgments across multiple quality dimensions, not just ROUGE against a single reference — is well-supported by our reproduction and remains a critical methodological principle for the field.

---

## References

Fabbri, A. R., Kryściński, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021). SummEval: Re-evaluating Summarization Evaluation. *Transactions of the Association for Computational Linguistics*, 9, 391–409.

Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Proceedings of the ACL Workshop on Text Summarization Branches Out*, 74–81.

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating Text Generation with BERT. *Proceedings of ICLR 2020*.

Nallapati, R., Zhou, B., Gulcehre, C., & Xiang, B. (2016). Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond. *CoNLL 2016*.

---

*Submitted to ReproNLP 2026 — Track A (Open Track)*  
*GEM 2026: 5th Generation, Evaluation & Metrics Workshop*
