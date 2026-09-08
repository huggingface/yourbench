import io
import base64
import hashlib
from pathlib import Path

import fitz
import trafilatura
from PIL import Image
from loguru import logger
from markitdown import MarkItDown

from datasets import Dataset
from huggingface_hub import InferenceClient
from yourbench.utils.dataset_engine import custom_save_dataset
from yourbench.utils.logging_context import log_step, log_stage, log_progress
from yourbench.utils.inference.inference_core import (
    InferenceCall,
    _load_models,
    run_inference,
)


def run(config) -> None:
    """Convert documents to markdown and optionally upload to Hub."""
    with log_stage(
        "ingestion",
        source_dir=str(config.pipeline.ingestion.source_documents_dir),
        output_dir=str(config.pipeline.ingestion.output_dir),
    ):
        ingestion_config = config.pipeline.ingestion
        source_dir = Path(ingestion_config.source_documents_dir)
        output_dir = Path(ingestion_config.output_dir)

        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        # Process files
        processor = _get_processor(config)
        successful_outputs: list[Path] = []

        # Collect all files to process
        all_files = sorted(
            f
            for f in source_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in ingestion_config.supported_file_extensions
        )
        logger.info(f"Found {len(all_files)} files to process")

        failures = []
        for idx, file_path in enumerate(all_files, 1):
            # Skip files in the output directory to prevent recursive processing
            try:
                if output_dir.resolve() in file_path.resolve().parents or file_path.resolve() == output_dir.resolve():
                    logger.debug(f"Skipping file in output directory: {file_path}")
                    continue
            except Exception:
                # If path resolution fails, skip the check
                pass

            log_progress(idx, len(all_files), f"file {file_path.name}")

            with log_step(f"converting_{file_path.name}"):
                try:
                    if content := _convert_file(file_path, config, processor):
                        # Preserve relative path to avoid filename collisions
                        relative_path = file_path.relative_to(source_dir)
                        output_path = output_dir / relative_path.with_name(relative_path.name + ".md")
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_text(content, encoding="utf-8")
                        logger.debug(f"Converted {file_path.name} → {output_path.name}")
                        successful_outputs.append(output_path)
                    else:
                        failures.append(file_path.name)
                except Exception:
                    failures.append(file_path.name)

        if failures:
            raise ValueError(f"Failed to ingest {len(failures)} document(s): {failures}")
        if not successful_outputs:
            raise ValueError("No supported nonempty documents were ingested")
        logger.info(f"Processed {len(successful_outputs)} files")

        # Save dataset locally and/or upload to Hub
        if successful_outputs:
            with log_step("uploading_to_hub"):
                _upload_to_hub(config, successful_outputs)


def _get_processor(config) -> MarkItDown:
    """Initialize markdown processor with optional LLM support."""
    if not config.pipeline.ingestion.llm_ingestion or not config.model_list:
        return MarkItDown()

    # Use the first model in the list for non-PDF LLM-based ingestion.
    model = config.model_list[0]
    try:
        client = InferenceClient(base_url=model.base_url, api_key=model.api_key)
        logger.debug(f"Using LLM for non-PDF ingestion: {model.model_name}")
        return MarkItDown(llm_client=client, llm_model=model.model_name)
    except Exception as e:
        logger.warning(f"Failed to init LLM processor: {e}")
        return MarkItDown()


def _convert_file(file_path: Path, config, processor: MarkItDown) -> str | None:
    """Convert file to markdown based on type."""
    ingestion_config = config.pipeline.ingestion
    supported_extensions = set(ingestion_config.supported_file_extensions)

    file_ext = file_path.suffix.lower()

    if file_ext not in supported_extensions:
        logger.warning(f"Unsupported file type: {file_ext} for file {file_path.name}")
        return None

    if file_ext == ".md":
        return file_path.read_text(encoding="utf-8")

    if file_ext in {".txt", ".text"}:
        return file_path.read_text(encoding="utf-8")

    if file_ext in {".html", ".htm"}:
        if content := _extract_html(file_path):
            return content
        # Fallback to MarkItDown
        return processor.convert(str(file_path)).text_content

    if file_ext == ".pdf" and config.pipeline.ingestion.llm_ingestion:
        content = _process_pdf_llm(file_path, config)
        if content is not None:
            return content
        # Fallback to standard conversion if LLM processing fails
        logger.warning(f"LLM PDF ingestion failed for {file_path.name}, falling back to standard conversion.")

    return processor.convert(str(file_path)).text_content


def _extract_html(path: Path) -> str | None:
    """Extract markdown from HTML using trafilatura."""
    try:
        html = path.read_text(encoding="utf-8")
        return trafilatura.extract(html, output_format="markdown", include_comments=False, include_tables=True)
    except Exception as e:
        logger.debug(f"Trafilatura failed for {path.name}: {e}")
        return None


def _process_pdf_llm(pdf_path: Path, config) -> str | None:
    """Convert every page of a PDF to Markdown using an LLM."""
    models = _load_models(config, "ingestion")
    ingestion_config = config.pipeline.ingestion

    if not models:
        logger.warning(f"No LLM models configured for PDF ingestion of {pdf_path.name}.")
        return None

    dpi = ingestion_config.pdf_dpi
    images = _pdf_to_images(pdf_path, dpi)
    if not images:
        return None  # Error already logged in _pdf_to_images

    prompt = ingestion_config.pdf_llm_prompt
    calls = [
        InferenceCall(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{_img_to_b64(img)}"},
                        },
                    ],
                }
            ],
            tags=["pdf_ingestion", f"page_{idx + 1}", pdf_path.name],
        )
        for idx, img in enumerate(images)
    ]

    pages: list[str] = []
    responses = run_inference(config, "ingestion", calls)
    if not responses:
        logger.error(f"LLM inference failed for all models on {pdf_path.name}")
        return None

    # Consolidate responses from all models
    for model_name in responses:
        pages.extend(responses[model_name])

    return "\n\n---\n\n".join(filter(None, pages))


def _pdf_to_images(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Convert PDF pages to images."""
    try:
        with fitz.open(pdf_path) as doc:
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                images.append(img)
            return images
    except Exception as e:
        logger.error(f"Failed to convert {pdf_path.name} to images: {e}")
        return []


def _img_to_b64(image: Image.Image) -> str:
    """Convert PIL image to base64."""
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


def _upload_to_hub(config, md_files: list[Path]):
    """Upload markdown files to Hugging Face Hub."""
    if not md_files:
        logger.warning("No markdown files to upload")
        return

    docs = []
    for path in md_files:
        try:
            if content := path.read_text(encoding="utf-8").strip():
                docs.append({
                    "document_id": hashlib.sha256(
                        (str(path.relative_to(Path(config.pipeline.ingestion.output_dir))) + "\0" + content).encode()
                    ).hexdigest()[:24],
                    "document_text": content,
                    "document_filename": path.name,
                    "document_metadata": {"file_size": path.stat().st_size},
                })
        except Exception as e:
            logger.error(f"Failed to read {path.name} for upload: {e}")

    if not docs:
        logger.warning("No valid documents to upload")
        return

    dataset = Dataset.from_list(docs)
    custom_save_dataset(dataset, config, subset="ingested", push_to_hub=config.hf_configuration.push_to_hub)
    logger.info(f"Uploaded {len(docs)} documents to Hub")
