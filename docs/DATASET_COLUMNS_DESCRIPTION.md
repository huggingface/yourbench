# Dataset subsets and columns

Each stage saves a named Hugging Face `Dataset` inside the local `DatasetDict`. With JSONL export enabled, each subset also has a file under `jsonl_export_dir`.

```python
from datasets import load_from_disk

subsets = load_from_disk("benchmark/datasets")
questions = subsets["prepared_lighteval"]
row = questions[0]
correct_choice = row["choices"][row["gold"][0]]
```

## Document subsets

| Subset | Row meaning | Fields |
| --- | --- | --- |
| `ingested` | One source document | `document_id`, `document_filename`, `document_text`, `document_metadata` |
| `summarized` | One source document with a summary | Ingested fields plus `document_summary`, `summarization_model` |
| `chunked` | One source document with all its chunks | Input document fields plus `chunks`, `multihop_chunks` |

`document_filename` is the original path relative to the source directory. Document IDs derive from that relative path and converted content: moving a corpus preserves IDs; changing content changes its ID. Treat IDs as opaque values.

`chunks` is a list of `{chunk_id, chunk_text}` objects. `multihop_chunks` is a list of `{chunk_ids, chunks_text}` groups. Summaries are present when chunking uses the `summarized` input subset.

## Question subsets

`single_hop_questions`, `multi_hop_questions`, and `cross_document_questions` contain one row per accepted question, rather than one row per document. Their common fields include:

| Field | Meaning |
| --- | --- |
| `question`, `answer` | Validated question and answer; MCQ answers are letters such as `B` |
| `self_answer` | Execution copy of the answer used by export |
| `question_mode` | `open-ended` or `multi-choice` |
| `choices` | MCQ options after deterministic shuffling, when applicable |
| `citations` | Source quotations supplied by the model, when required by the schema |
| `question_data` | Original schema-validated payload before MCQ shuffling |
| `sources` | Canonical list of `{document_id, chunk_id}` references |
| `generating_model` | Model that generated this question |
| `raw_response` | Complete generation response containing the question |
| `additional_instructions` | Stage instructions used for generation |

Default schemas also provide fields such as `question_type`, `estimated_difficulty` and `thought_process`. Custom fields retain their declared names and values; difficulty words are not converted into numbers. `self_assessed_question_type` is derived from `question_type` for export compatibility.

Single-hop rows expose `chunk_id`; multi-hop rows expose `source_chunk_ids`. Use `sources` for provenance, especially across documents; do not decode synthetic IDs. Question rows do not automatically contain all document text and chunk structures.

Rewriting writes separate `*_rewritten` subsets. It preserves answers, choices, custom fields, source references and the original `question_data`, while replacing `question` and adding `original_question`, `question_rewriting_model`, `question_rewriting_rationale`, and `raw_question_rewriting_response`. The original generation subset remains available.

## Evaluation subset

`prepare_lighteval` combines selected question subsets into `prepared_lighteval` by default. It preserves question fields and resolves their sources, adding:

| Field | Meaning |
| --- | --- |
| `kind` | `single_hop`, `multi_hop`, or `cross_document` |
| `ground_truth_answer` | Answer text for open-ended questions; answer letter for MCQs |
| `choices`, `gold` | Answer choices and zero-based indices of the correct choices |
| `question_category` | Question type from generation |
| `question_generating_model` | Generating model identifier |
| `chunk_ids`, `chunks` | Referenced chunk IDs and resolved text, in source-reference order |
| `document_ids`, `documents` | Unique referenced document IDs and their full text, in first-reference order |
| `document` | All referenced document texts joined with blank lines |
| `document_summary` | All referenced document summaries joined with blank lines |

For open-ended questions, `choices` is `[answer_text]` and `gold` is `[0]`. For MCQs, `gold` indexes the shuffled `choices`. Do not use the original answer in `question_data` to index shuffled choices.

Exports fail on unresolved source references. Additional custom fields are retained across rows when their types are compatible in Arrow; incompatible combined schemas fail explicitly. See [custom schemas](CUSTOM_SCHEMAS.md).

## Citation scores

`citation_score_filtering` adds three columns to its selected subset without dropping rows:

- `answer_citation_score`: mean fuzzy overlap between each citation and the answer.
- `chunk_citation_score`: mean of each citation's best fuzzy overlap with a source chunk.
- `citation_score`: `alpha * chunk_citation_score + beta * answer_citation_score`.

The component scores range from 0 to 100. Default weights are 0.7 and 0.3. These are lexical overlap measures, not correctness judgments or semantic entailment scores; MCQ answer letters are especially weak signals for answer overlap.
