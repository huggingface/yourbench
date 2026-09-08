# Minimal pipeline

This recipe reads the included fictional policy documents and runs ingestion, summarization, token chunking, single-hop question generation and evaluation export.

Set the endpoint variables described in [the examples guide](../README.md), then run from the repository root:

```bash
yourbench run example/default_example/config.yaml
```

Inspect `example/default_example/output/jsonl/prepared_lighteval.jsonl`. The model determines the number of accepted questions; the example does not promise a fixed count. Outputs stay local.
