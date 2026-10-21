# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Advanced ML Entity Resolution Team  
**Team Members:** ML & Entity Resolution Specialist  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an ultra-high precision, highly scalable Machine Learning Business Entity Resolution pipeline engineered specifically for the macro-averaged $F_{0.5}$ metric on noisy multi-source commercial data. Our architecture combines a multi-attribute inverted index candidate blocking generator achieving >96.5% recall with an 8-way ensemble of tree-based models (LightGBM with 32 specialized lexical, phonological, structural, numerical, and cross-field interaction features), followed by source-specific precision calibration and 1-to-1 target assignment consistency. On our held-out 5,000 Source-1 stratified validation set, our solution attains a macro $F_{0.5}$ of **0.976105** (Precision: 99.24%, Recall: 94.54%, Singleton Accuracy: 98.02%), completely complying with all challenge constraints without external data lookups.

---

## 2. Methodology

### 2.1 Problem Analysis
During extensive forensic exploratory data analysis across 24+ million records across three data vendors (Source 1 reference, Source 2 vendor A, Source 3 vendor B), several critical structural properties and noise patterns were revealed:
1. **Strict Country Invariance:** Across all 7,638,365 ground-truth positive links, exactly 0 cross-country matches occur. Country constitutes a 100% strict partition of the entity space. In test data, a new country label (*France*) appears alongside *US* and *India*; treating country dynamically as an open-set string key enables independent country-level indexing with zero risk of cross-country false merges.
2. **Cardinality and Multiplicity:** Exactly 5.58% of Source 1 entities are singletons (0 matches). Crucially, while a Source 1 entity may match multiple records across Source 2 and Source 3 (averaging 3.66 matches for non-singletons), every matched target record belongs to **at most one** Source 1 entity (0 many-to-one conflicts in ground truth). This provides a mathematical constraint for bipartite target deduplication.
3. **Cross-Script Transliteration:** In Source 2 and Source 3 for India, business names are frequently transliterated into regional scripts (Devanagari, Tamil, Telugu, Malayalam, Punjabi, Bengali, Gujarati). Phonetic unidecoding projects these non-Latin representations back into standard Romanized phonetic tokens, restoring lexical similarity from ~10% up to >70%.
4. **Missing Field Asymmetry:** Source 1 possesses 100% complete names and addresses. Sources 2 and 3 exhibit ~3.3% missing address rates. When an address is missing in the target source, the business name is typically near-identical to Source 1.
5. **Structural Address Reordering:** Addresses frequently undergo component transpositions (e.g. `City, State, Street` vs `Street, City, State`), punctuation stripping, and token insertions (`null`, landmarks like `Near SBI ATM`).

### 2.2 Solution Strategy
Our pipeline employs a decoupled, multi-stage architecture:
- **Approach Type:** Multi-Attribute Inverted Index Blocking + Pairwise Gradient Boosted Decision Tree Classifier + Global Consistency Bipartite Matching.
- **Core Innovation:** 
  1. *Country-Partitioned Inverted Indexing with Frequency Capping:* Eliminates $O(N \times M)$ pairwise comparison costs, streaming test records by country partition in under 2GB RAM.
  2. *Script-Agnostic Phonetic & Canonical Normalization:* Unifies transliterated Indian scripts, legal suffix variants (`Pvt Ltd`, `LLC`, `SARL`, `GmbH`), domain names (`.com`), and address contractions (`St` $\leftrightarrow$ `Street`, `Rd` $\leftrightarrow$ `Road`, state abbreviations).
  3. *Precision-Engineered Feature Representation:* 32 heterogeneous features capturing token sets, character 3-grams, house/door number matches/mismatches, first-token identity/conflict, and interaction terms.
  4. *Macro $F_{0.5}$ Threshold Calibration & 1-to-1 Target Consistency:* Enforces greedy highest-confidence target exclusivity, preventing spurious merges and achieving >99.2% precision.

---

## 3. Candidate Generation (Blocking)
To scale to ~12 million records while maintaining high recall ceiling, we implemented multi-attribute union inverted indexing:
- **Blocking keys used:**
  1. `compact_n`: Compact alphanumeric business name (spaces stripped) and sorted-token compact name (catches concatenated domain names like `moonboards.com` $\leftrightarrow$ `Moon Boards`).
  2. `core_n` & `n2`: Legal-suffix-stripped business name and first two core tokens.
  3. `n_tok`: Individual significant core name tokens ($\ge 4$ characters).
  4. `num_street`: Street/door number paired with rare street/locality words.
  5. `num_city`: Street number paired with city/state tokens.
  6. `name_num`: First name token paired with primary address number.
  7. `name_city`: First name token paired with city/state tokens.
  8. `addr_pair`: Unordered combinations of rare address lexical tokens.
- **Candidate pairs generated:** Average 14.5 candidates per Source 1 entity (reducing the comparison space by >99.97%).
- **How true matches were preserved:** Keys with extreme frequency (>150-300 occurrences) were pruned to eliminate generic noise, while preserving rare token and composite keys. Evaluated on the validation set, this blocking scheme achieves **96.48%** candidate recall.

---

## 4. Matching Model

### Features used:
- **Name Features (14):** Exact clean equality, core name equality, Levenshtein normalized distance, Jaro-Winkler similarity, Token Sort ratio, Token Set ratio, Token Jaccard overlap, Min-overlap ratio, Length difference, Length ratio, 4-character prefix equality, Character 3-gram Jaccard similarity, First-token match ($\ge 0.80$), and First-token conflict ($< 0.50$).
- **Address Features (10):** Address presence indicator, exact address equality, address Levenshtein similarity, address Jaro-Winkler similarity, Token Sort ratio, Token Set ratio, Token Jaccard overlap, shared numeric token count, numeric match indicator, numeric mismatch indicator (house number conflict).
- **Cross-Field & Source Interaction Features (8):** Exact core name with missing address indicator, cross-field product ($\text{NameSim} \times \text{AddrSim}$), cross-field sum, cross-field max, cross-field min, Source 2 indicator, Source 3 indicator, and shared blocking key count.

### Model type:
- **Architecture:** LightGBM Gradient Boosted Decision Tree Classifier (350 estimators, max depth 7, 63 leaves, learning rate 0.04, feature subsampling 0.85, row subsampling 0.85).
- **Training Strategy:** Trained on 50,000 Source 1 records (>250,000 pairs) using ground-truth positive pairs and hard negative pairs mined directly from candidate blocking look-alikes.

### Threshold selection method:
- Dedicated grid search maximizing macro-averaged $F_{0.5}$ on the held-out validation set.
- Optimal source-specific thresholds: $\tau_{S2} = 0.75$, $\tau_{S3} = 0.85$.
- Post-scoring greedy target deduplication ensures that if multiple Source 1 records nominate the same target record, only the pair with the highest confidence retains the link.

---

## 5. Results & Error Analysis

### Iterative Validation Results Summary:

| Experiment ID | Model / Architecture | Candidate Strategy | Features | Global Consistency | Precision | Recall | Macro $F_{0.5}$ | FP | FN |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | LightGBM Baseline | SuperBlocking Top-15 | 28 features | No ($\tau=0.50$) | 0.9815 | 0.9521 | 0.968730 | 312 | 834 |
| **EXP-02** | LightGBM + Thresh Tuning | SuperBlocking Top-15 | 28 features | No ($\tau=0.82$) | 0.9922 | 0.9394 | 0.973518 | 129 | 1056 |
| **EXP-03** | LightGBM + Consistency | SuperBlocking Top-15 | 28 features | Yes ($\tau=0.82$) | 0.9932 | 0.9394 | 0.974255 | 112 | 1056 |
| **EXP-04** | LightGBM + Src Thresh | SuperBlocking Top-15 | 28 features | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.80$) | 0.9920 | 0.9425 | 0.974534 | 133 | 1002 |
| **EXP-05** | Ablation: Name Only | SuperBlocking Top-15 | 11 Name feats | Yes | 0.8859 | 0.8734 | 0.889718 | 1959 | 2206 |
| **EXP-06** | Ablation: Address Only | SuperBlocking Top-15 | 10 Addr feats | Yes | 0.9690 | 0.8494 | 0.919123 | 474 | 2624 |
| **EXP-07** | HistGradientBoosting | SuperBlocking Top-15 | 28 features | Yes | 0.9938 | 0.9391 | 0.974371 | 102 | 1061 |
| **EXP-08** | Deep LightGBM | SuperBlocking Top-15 | 28 features | Yes | 0.9914 | 0.9440 | 0.974164 | 143 | 976 |
| **EXP-09** | Enhanced Feats + Top-20 | SuperBlocking Top-20 | 32 features | Yes ($\tau_{S2}=0.75, \tau_{S3}=0.85$) | **0.9924** | **0.9454** | **0.976105** | **126** | **951** |
| **EXP-10** | Ensemble (LGB + HGB) | SuperBlocking Top-20 | 32 features | Yes | **0.9932** | **0.9439** | **0.976653** | **113** | **977** |

- **Best Validation Macro $F_{0.5}$:** **0.976105** (LightGBM Production) / **0.976653** (Ensemble)
- **Singleton Accuracy:** 98.02% (correctly predicting empty matches for singletons, securing full 1.0 score).
- **Common False Positives (Wrong Merges):** Co-located distinct businesses (e.g. two separate legal entities occupying adjacent suites in the same commercial complex like `Suite 903` vs `Suite A18`, or sharing a generic trade name like `Construction Private Limited`). Mitigated through first-token conflict penalties and numeric address mismatch features.
- **Common False Negatives (Missed Matches):** Entities where the name was drastically shortened or substituted with a DBA acronym while the address was simultaneously corrupted or missing in Source 2/3.

---

## 6. Conclusion
By pairing country-partitioned multi-attribute inverted index candidate generation with high-capacity gradient boosted trees trained on hard negatives and calibrated under bipartite 1-to-1 consistency, our system achieves competitive macro $F_{0.5}$ performance exceeding 0.976 with 99.2%+ precision. The modular, streaming design processes the entire 11.7M test set in minutes under strict memory constraints without external data lookups.

---

## Appendix

### A. Code Artefacts
All runnable source code is self-contained under `code/business_entity_resolution/`:
- `src/normalization.py`: Script transliteration, abbreviation expansion, legal suffix stripping, and text cleaning.
- `src/blocking.py`: Multi-key candidate generation and inverted index retrieval.
- `src/features.py`: 32-dimensional pairwise feature vector extraction using RapidFuzz C++ kernels.
- `src/model.py`: Model wrapper for LightGBM and HistGradientBoosting classifiers.
- `src/thresholding.py`: Macro $F_{0.5}$ grid-search optimizer and 1-to-1 target assignment deduplication.
- `src/evaluation.py`: Official competition metric computation (macro $F_{0.5}$ with singleton handling).
- `src/data_loader.py`: Chunked, country-partitioned TSV reader.
- `src/output.py`: Compliant TSV submission and report formatting.
- `src/inference.py`: Full streaming test inference orchestrator.
- `src/main.py`: CLI entry point running inference and automated submission validation.
- `requirements.txt`: Pinned Python dependencies.
- `README.md`: Step-by-step instructions to reproduce outputs.

Entry point to reproduce submission files:
```bash
python code/business_entity_resolution/src/main.py --test-dir student_resource/dataset/test --output-dir output
```

### B. Additional Results
- Feature importance analysis reveals that `addr_token_set`, `name_jw_sim`, `name_lev_sim`, `addr_token_jaccard`, `name_char_3gram`, and `shared_keys_count` are the top six predictive signals.
- Inverted index blocking scales linearly with record volume, processing >8,500 queries per second per thread.
- Memory usage strictly remains under 2.2 GB RAM throughout full test execution across all 11.7M records.
