"""Tests for citation_score_filtering pipeline stage."""

from unittest.mock import Mock, patch

import pytest

from datasets import Dataset
from yourbench.pipeline.citation_score_filtering import CitationScoreCalculator, run


class TestCitationScoreCalculator:
    """Tests for CitationScoreCalculator."""

    def test_compute_empty_citations(self):
        """Empty citations returns zeros."""
        calc = CitationScoreCalculator(alpha=0.5, beta=0.5)
        result = calc.compute(citations=[], chunks=["text"], answer="answer")
        assert result == (0.0, 0.0, 0.0)

    def test_compute_empty_chunks_and_answer(self):
        """Empty chunks and empty answer returns zeros."""
        calc = CitationScoreCalculator(alpha=0.5, beta=0.5)
        result = calc.compute(citations=["citation"], chunks=[], answer="")
        assert result == (0.0, 0.0, 0.0)

    def test_compute_basic(self):
        """Basic citation score calculation."""
        calc = CitationScoreCalculator(alpha=0.5, beta=0.5)
        result = calc.compute(
            citations=["test content"],
            chunks=["test content chunk"],
            answer="test content answer",
        )
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
        assert all(v >= 0 for v in result)

    def test_compute_weighted_alpha_beta(self):
        """Verify alpha/beta weighting affects final score."""
        calc1 = CitationScoreCalculator(alpha=1.0, beta=0.0)
        calc2 = CitationScoreCalculator(alpha=0.0, beta=1.0)

        citations = ["exact match"]
        chunks = ["exact match"]
        answer = "different text"

        result1 = calc1.compute(citations, chunks, answer)
        result2 = calc2.compute(citations, chunks, answer)

        # alpha=1 weights chunk score fully, beta=1 weights answer score fully
        # Chunk score should be higher (exact match vs different text)
        assert result1[2] > result2[2]


class TestCitationScoreRun:
    """Tests for the run() function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock()
        config.pipeline.citation_score_filtering.run = True
        config.pipeline.citation_score_filtering.subset = "test_subset"
        config.pipeline.citation_score_filtering.alpha = 0.5
        config.pipeline.citation_score_filtering.beta = 0.5
        config.hf_configuration.push_to_hub = False
        return config

    def test_run_disabled_skips_processing(self, mock_config):
        """When run=False, stage is skipped."""
        mock_config.pipeline.citation_score_filtering.run = False

        with patch("yourbench.pipeline.citation_score_filtering.custom_load_dataset") as mock_load:
            run(mock_config)
            mock_load.assert_not_called()

    def test_run_empty_dataset_fails(self, mock_config):
        """Empty dataset triggers warning and early return."""
        with patch("yourbench.pipeline.citation_score_filtering.custom_load_dataset") as mock_load:
            mock_load.return_value = Dataset.from_dict({})

            with patch("yourbench.pipeline.citation_score_filtering.custom_save_dataset") as mock_save:
                with pytest.raises(ValueError, match="empty"):
                    run(mock_config)
                mock_save.assert_not_called()

    def test_run_processes_rows(self, mock_config):
        """Verify rows are processed and scores computed."""
        test_data = {
            "citations": [["citation1"]],
            "chunks": [["chunk1"]],
            "ground_truth_answer": ["answer1"],
        }

        with patch("yourbench.pipeline.citation_score_filtering.custom_load_dataset") as mock_load:
            mock_load.return_value = Dataset.from_dict(test_data)

            with patch("yourbench.pipeline.citation_score_filtering.custom_save_dataset") as mock_save:
                with patch("yourbench.pipeline.citation_score_filtering.replace_dataset_columns") as mock_replace:
                    mock_replace.return_value = Dataset.from_dict(test_data)
                    run(mock_config)

                    # Check replace_dataset_columns was called with score columns
                    mock_replace.assert_called_once()
                    args = mock_replace.call_args
                    columns = args[0][1]  # Second positional arg is columns_data
                    assert "answer_citation_score" in columns
                    assert "chunk_citation_score" in columns
                    assert "citation_score" in columns
