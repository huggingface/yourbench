"""Convert each source exactly once; publish a dataset only after all inputs succeed."""

import io
import base64
import hashlib
from pathlib import Path

import fitz
import trafilatura
from PIL import Image
from markitdown import MarkItDown

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_save_dataset
from yourbench.utils.inference.inference_core import InferenceCall, _load_models, run_inference


def source_files(config) -> list[Path]:
    """Deterministic source discovery shared by preflight and ingestion."""
    source = Path(config.source_documents_dir).resolve()
    output = Path(config.output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    extensions = {extension.lower() for extension in config.supported_file_extensions}
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and not path.resolve().is_relative_to(output)
    )
    if not files:
        raise ValueError("No supported source documents found")
    return files


def run(config) -> None:
    cfg = config.pipeline.ingestion
    paths = source_files(cfg)
    source, output = Path(cfg.source_documents_dir).resolve(), Path(cfg.output_dir).resolve()
    processor = MarkItDown()
    documents = []
    for path in paths:
        content = _convert_file(path, config, processor).strip()
        if not content:
            raise ValueError(f"Document produced no text: {path.relative_to(source)}")
        relative = path.relative_to(source)
        destination = output / relative.with_name(relative.name + ".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        documents.append({
            "document_id": hashlib.sha256((relative.as_posix() + "\0" + content).encode()).hexdigest()[:24],
            "document_filename": relative.as_posix(),
            "document_text": content,
            "document_metadata": {"file_size": path.stat().st_size},
        })
    custom_save_dataset(
        Dataset.from_list(documents), config, subset="ingested", push_to_hub=config.hf_configuration.push_to_hub
    )


def _convert_file(path: Path, config, processor: MarkItDown) -> str:
    match path.suffix.lower():
        case ".md" | ".txt" | ".text":
            return path.read_text(encoding="utf-8")
        case ".html" | ".htm":
            text = trafilatura.extract(
                path.read_text(encoding="utf-8"), output_format="markdown", include_comments=False, include_tables=True
            )
            return text or processor.convert(str(path)).text_content
        case ".pdf" if config.pipeline.ingestion.llm_ingestion:
            return _process_pdf_llm(path, config)
        case _:
            return processor.convert(str(path)).text_content


def _process_pdf_llm(pdf_path: Path, config) -> str:
    if len(_load_models(config, "ingestion")) != 1:
        raise ValueError("PDF ingestion requires exactly one model")
    cfg = config.pipeline.ingestion
    images = _pdf_to_images(pdf_path, cfg.pdf_dpi)
    if not images:
        raise ValueError(f"PDF has no pages: {pdf_path.name}")
    calls = [
        InferenceCall(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cfg.pdf_llm_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_img_to_b64(image)}"}},
                    ],
                }
            ],
            tags=["pdf_ingestion", f"page_{index + 1}"],
        )
        for index, image in enumerate(images)
    ]
    responses = run_inference(config, "ingestion", calls)
    pages = next(iter(responses.values()))
    if len(pages) != len(calls) or any(not page.strip() for page in pages):
        raise ValueError("PDF extraction returned incomplete page text")
    return "\n\n---\n\n".join(pages)


def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    with fitz.open(pdf_path) as document:
        pixmaps = [page.get_pixmap(dpi=dpi, alpha=False) for page in document]
    return [Image.frombytes("RGB", (pix.width, pix.height), pix.samples) for pix in pixmaps]


def _img_to_b64(image: Image.Image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
