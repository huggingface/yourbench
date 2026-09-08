# Local inference

Serve a chat-completion-compatible model at `http://localhost:8000/v1`, set `YOURBENCH_MODEL` to its served name, then run:

```bash
yourbench run example/local_vllm_private_data/config.yaml
```

This recipe uses the included Markdown policy corpus. It saves locally and does not require a Hub token. If your server requires authentication, add `api_key: $YOURBENCH_API_KEY` and set that variable. Change `base_url` if your server uses a different address.

The inference client may fall back to `HF_TOKEN` when `api_key` is omitted. Only this recipe's configured local endpoint receives model requests; dataset publication is disabled explicitly.

To use HTML documents, change `source_documents_dir` and set `supported_file_extensions: [.html]`. Cross-document generation needs at least two source documents. See [endpoint configuration](../../docs/USING_OPENAI_COMPATIBLE_MODELS.md).
