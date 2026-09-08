"""Integration tests for pipeline stages."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from datasets import Dataset


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock config using OmegaConf."""
    return OmegaConf.create({
        "hf_configuration": {
            "hf_token": "fake_token",
            "hf_organization": "fake_org",
            "private": True,
            "hf_dataset_name": "fake_dataset",
            "concat_if_exist": False,
            "local_dataset_dir": temp_dir,
            "local_saving": True,
            "upload_card": False,
            "push_to_hub": False,
        },
        "model_list": [
            {
                "model_name": "fake_model",
                "provider": None,
                "api_key": "fake_key",
                "base_url": "http://localhost:8000/v1",
                "max_concurrent_requests": 1,
                "encoding_name": "cl100k_base",
                "bill_to": None,
                "extra_parameters": {},
            }
        ],
        "model_roles": {
            "ingestion": ["fake_model"],
            "summarization": ["fake_model"],
            "chunking": ["fake_model"],
            "single_hop_question_generation": ["fake_model"],
            "multi_hop_question_generation": ["fake_model"],
        },
        "pipeline": {
            "ingestion": {
                "run": True,
                "source_documents_dir": os.path.join(temp_dir, "raw"),
                "output_dir": os.path.join(temp_dir, "processed"),
                "llm_ingestion": False,
                "pdf_dpi": 300,
                "supported_file_extensions": [".md", ".txt", ".pdf"],
            },
            "summarization": {
                "run": True,
                "max_tokens": 32768,
                "token_overlap": 512,
                "encoding_name": "cl100k_base",
                "summarization_user_prompt": "",
                "combine_summaries_user_prompt": "",
            },
            "chunking": {
                "run": True,
                "l_max_tokens": 256,
                "h_min": 2,
                "h_max": 5,
                "num_multihops_factor": 2,
                "token_overlap": 32,
                "encoding_name": "cl100k_base",
            },
            "single_hop_question_generation": {
                "run": True,
                "question_mode": "open-ended",
                "additional_instructions": "Generate questions to test a curious adult",
                "single_hop_system_prompt": "You are a helpful assistant.",
                "single_hop_system_prompt_multi": "You are a helpful assistant.",
                "single_hop_user_prompt": "",
                "chunk_sampling": {
                    "enable": False,
                    "num_samples": 100,
                    "strategy": "random",
                    "random_seed": 42,
                },
            },
            "multi_hop_question_generation": {
                "run": True,
                "question_mode": "multi-choice",
                "additional_instructions": "Generate multi-choice questions",
                "multi_hop_system_prompt": "You are a helpful assistant.",
                "multi_hop_system_prompt_multi": "You are a helpful assistant.",
                "multi_hop_user_prompt": "",
            },
            "cross_document_question_generation": {"run": False},
            "question_rewriting": {"run": False},
            "prepare_lighteval": {
                "run": True,
                "single_hop_subset": "single_hop_questions",
                "multi_hop_subset": "multi_hop_questions",
                "cross_doc_subset": "cross_document_questions",
                "chunked_subset": "chunked",
                "summarized_subset": "summarized",
                "output_subset": "prepared_lighteval",
            },
            "lighteval": {"run": False},
            "citation_score_filtering": {"run": False},
        },
        "debug": False,
    })


@pytest.mark.parametrize("mock_no_docs", [False, True])
def test_ingestion_stage(mock_config, temp_dir, mock_no_docs):
    """Test the ingestion stage."""
    raw_dir = mock_config.pipeline.ingestion.source_documents_dir
    output_dir = mock_config.pipeline.ingestion.output_dir
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not mock_no_docs:
        with open(os.path.join(raw_dir, "test_doc.txt"), "w") as f:
            f.write("This is a test document for ingestion.")

    with (
        patch("yourbench.pipeline.ingestion._convert_file") as mock_convert,
        patch("yourbench.pipeline.ingestion.custom_save_dataset") as mock_save,
    ):
        mock_convert.return_value = "mocked content"
        from yourbench.pipeline.ingestion import run

        if mock_no_docs:
            with pytest.raises(ValueError, match="No supported"):
                run(mock_config)
        else:
            run(mock_config)

        if mock_no_docs:
            mock_convert.assert_not_called()
            mock_save.assert_not_called()
        else:
            mock_convert.assert_called()
            mock_save.assert_called_once()


def test_summarization_stage(mock_config):
    """Test the summarization stage."""
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1", "doc2"],
        "document_text": ["This is document 1", "This is document 2"],
        "document_filename": ["doc1.md", "doc2.md"],
    })

    with (
        patch("yourbench.pipeline.summarization.custom_load_dataset", return_value=mock_dataset),
        patch("yourbench.pipeline.summarization.custom_save_dataset") as mock_save,
        patch("yourbench.pipeline.summarization.run_inference") as mock_run_inference,
    ):
        mock_run_inference.return_value = {
            "fake_model": [
                '{"summary":"Summary for doc1"}',
                '{"summary":"Summary for doc2"}',
            ]
        }
        from yourbench.pipeline.summarization import run

        run(mock_config)

        assert mock_run_inference.call_count == 1
        mock_save.assert_called_once()


def test_chunking_stage(mock_config):
    """Test the chunking stage."""
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1", "doc2"],
        "document_text": [
            "This is document 1 with enough text to be chunked properly. " * 10,
            "This is document 2 which also has sufficient text for chunking. " * 10,
        ],
        "document_summary": ["Summary 1", "Summary 2"],
    })

    with (
        patch("yourbench.pipeline.chunking.custom_load_dataset", return_value=mock_dataset),
        patch("yourbench.pipeline.chunking.custom_save_dataset") as mock_save,
        patch("yourbench.pipeline.chunking.split_into_token_chunks") as mock_split,
    ):
        mock_split.return_value = ["Chunk 1", "Chunk 2"]
        from yourbench.pipeline.chunking import run

        run(mock_config)

        assert mock_split.call_count == 2
        mock_save.assert_called_once()
        assert mock_save.call_args[1]["subset"] == "chunked"


def test_single_hop_question_generation_stage(mock_config):
    """Test the single-shot question generation stage."""
    chunks = [{"chunk_id": "chunk1", "chunk_text": "This is chunk 1"}]
    mock_dataset = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_summary": ["Document 1 summary"],
        "document_filename": ["doc1.md"],
        "chunks": [chunks],
    })

    with (
        patch("yourbench.pipeline.question_generation._core.custom_load_dataset", return_value=mock_dataset),
        patch("yourbench.pipeline.question_generation._core.custom_save_dataset") as mock_save,
        patch("yourbench.pipeline.question_generation._core.run_inference") as mock_run_inference,
        patch("yourbench.utils.parsing_engine.parse_qa_pairs_from_response") as mock_parse,
    ):
        mock_run_inference.return_value = {"fake_model": ["Question generation response"]}
        mock_parse.return_value = [
            {
                "question": "Test question?",
                "answer": "Test answer",
                "estimated_difficulty": 5,
                "question_type": "factual",
                "question_mode": "open-ended",
                "thought_process": "Reasoning",
                "citations": ["citation"],
            }
        ]

        from yourbench.pipeline.question_generation import run_single_hop

        run_single_hop(mock_config)

        mock_run_inference.assert_called_once()
        mock_parse.assert_called_once()
        mock_save.assert_called_once()


def test_lighteval_stage(mock_config):
    """Test the lighteval stage."""
    single_hop_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "chunk_id": ["chunk1"],
        "question": ["Single-shot question?"],
        "self_answer": ["Single-shot answer"],
        "estimated_difficulty": [5],
        "self_assessed_question_type": ["factual"],
        "question_mode": ["open-ended"],
        "generating_model": ["fake_model"],
        "additional_instructions": ["Generate questions"],
    })

    multi_hop_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "source_chunk_ids": [["chunk1", "chunk2"]],
        "question": ["Multi-hop question?"],
        "self_answer": ["A"],
        "choices": [["(A) First", "(B) Second", "(C) Third", "(D) Fourth"]],
        "estimated_difficulty": [7],
        "self_assessed_question_type": ["reasoning"],
        "question_mode": ["multi-choice"],
        "generating_model": ["fake_model"],
        "additional_instructions": ["Generate questions"],
    })

    chunked_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_text": ["Full document text"],
        "chunks": [
            [
                {"chunk_id": "chunk1", "chunk_text": "Chunk 1 text"},
                {"chunk_id": "chunk2", "chunk_text": "Chunk 2 text"},
            ]
        ],
    })

    summarized_ds = Dataset.from_dict({
        "document_id": ["doc1"],
        "document_summary": ["Document 1 summary"],
    })

    with (
        patch("yourbench.pipeline.prepare_lighteval.custom_load_dataset") as mock_load,
        patch("yourbench.pipeline.prepare_lighteval.custom_save_dataset") as mock_save,
        patch("datasets.Dataset.from_list") as mock_from_list,
    ):

        def load_dataset_side_effect(config, subset):
            return {
                "single_hop_questions": single_hop_ds,
                "multi_hop_questions": multi_hop_ds,
                "chunked": chunked_ds,
                "summarized": summarized_ds,
                "cross_document_questions": Dataset.from_dict({}),
            }.get(subset, Dataset.from_dict({}))

        mock_load.side_effect = load_dataset_side_effect
        mock_from_list.return_value = Dataset.from_dict({
            "question": ["Combined question 1", "Combined question 2"],
            "ground_truth_answer": ["Answer 1", "Answer 2"],
        })

        from yourbench.pipeline.prepare_lighteval import run

        run(mock_config)

        assert mock_load.call_count == 5
        mock_from_list.assert_called_once()
        mock_save.assert_called_once()
