# PDF page-image ingestion

This directory retains its historical Gemini example name. Set the endpoint variables from [the examples guide](../README.md) to any accessible model that supports images in chat requests.

```bash
yourbench run example/rich_pdf_extraction_with_gemini/config.yaml
```

The recipe renders the included small PDF page as an image and asks the model to transcribe it before summarization and question generation. `llm_ingestion: true` applies to PDFs only; exactly one model must be assigned to ingestion. Page-rendering and model errors stop the stage rather than silently switching extraction methods.

Replace `source_documents_dir` with your own PDFs to try tables, figures or equations. Vision extraction can still omit or misread details: compare the generated `output/processed/` text with the source PDF. The checked-in fixture is intentionally plain text and does not establish chart-extraction quality. Results are saved locally under `output/`.
