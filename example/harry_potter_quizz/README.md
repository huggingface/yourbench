# Multiple-choice quiz

This directory retains its historical tutorial name. The checked-in recipe now uses the included fictional policy corpus, so no book download is needed. Point `ingestion.source_documents_dir` at your own story documents to create a literary quiz.

Set the endpoint variables from [the examples guide](../README.md), then run:

```bash
yourbench run example/harry_potter_quizz/config.yaml
```

The recipe enables `question_mode: multi-choice` and `prepare_lighteval`. Results are in `output/jsonl/prepared_lighteval.jsonl` beside the recipe. `gold` contains zero-based indices into the shuffled `choices`; `ground_truth_answer` contains the corresponding letter. Review distractors and source passages before using the quiz.

See [dataset columns](../../docs/DATASET_COLUMNS_DESCRIPTION.md) for the complete output contract.
