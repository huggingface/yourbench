"""Tests for question rewriting pipeline fixes from PR #181.

These tests verify:
1. STAGE_TAG is used correctly (not wrapped in extra list)
2. QuestionRow validation handles missing question_mode field
3. Graceful handling when multi_hop_questions subset doesn't exist
"""

import unittest
from unittest.mock import Mock, patch

from datasets import Dataset
from yourbench.pipeline.question_rewriting import (
    STAGE_TAG,
    _process_question_type,
    _build_question_rewriting_calls,
    _process_question_rewriting_responses,
)


class TestQuestionRewritingFixes(unittest.TestCase):
    """Test cases for PR #181 fixes."""

    def test_stage_tag_is_flat_list(self):
        """Test that STAGE_TAG is already a list and not wrapped again."""
        self.assertEqual(STAGE_TAG, ["question_rewriting"])

    def test_inference_call_receives_flat_tags(self):
        """Test that InferenceCall receives tags as flat list, not nested.

        This test verifies the fix for issue #178 where tags=[STAGE_TAG] created
        a nested list [["question_rewriting"]] instead of ["question_rewriting"],
        causing "sequence item 0: expected str instance, list found" error.
        """
        dataset = Dataset.from_list([
            {
                "question": "What is the capital?",
                "chunks": "Paris is the capital.",
                "document_summary": "Summary",
                "self_answer": "Paris",
            }
        ])

        calls, indices = _build_question_rewriting_calls(
            dataset=dataset,
            system_prompt="System prompt",
            user_prompt_template="Q: {original_question}\nA: {answer}\nChunk: {chunk_text}\nSummary: {document_summary}\n{additional_instructions}",
            additional_instructions="Rewrite the question",
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(indices), 1)
        # This is the critical fix - tags should be ["question_rewriting"], not [["question_rewriting"]]
        self.assertEqual(calls[0].tags, ["question_rewriting"])
        self.assertIsInstance(calls[0].tags, list)
        self.assertEqual(len(calls[0].tags), 1)
        self.assertIsInstance(calls[0].tags[0], str)

    def test_question_mode_default_for_missing_field(self):
        """Test that question_mode defaults to 'open-ended' when missing from dataset.

        This test verifies the fix where older datasets don't have question_mode field,
        causing QuestionRow validation to fail. The fix adds a default value before
        creating the QuestionRow object.
        """
        responses = {"model-1": ["<rewritten_question>What is the capital city?</rewritten_question>"]}
        indices = [0]

        # Original dataset WITHOUT question_mode field (simulating old datasets)
        # Include all required QuestionRow fields
        original_dataset = Dataset.from_list([
            {
                "document_id": "doc1",
                "additional_instructions": "Test instructions",
                "question": "What is capital?",
                "self_answer": "Paris",
                "estimated_difficulty": 5,
                "self_assessed_question_type": "factual",
                "generating_model": "test-model",
                "thought_process": "Test thought",
                "raw_response": "Test response",
                "chunk_id": "chunk1",  # Required - either chunk_id or source_chunk_ids
                # Note: question_mode is missing - this is the bug we're fixing
            }
        ])

        # Process responses - should not raise validation error
        rewritten_rows = _process_question_rewriting_responses(responses, indices, original_dataset)

        # Verify we got a row back (the fix allows this to succeed)
        self.assertEqual(len(rewritten_rows), 1, "Should successfully process row even without question_mode")

        # Verify the question was actually rewritten
        self.assertEqual(rewritten_rows[0]["question"], "What is the capital city?")

        # Note: question_mode is NOT in to_dict() output, but it was used during validation
        # The important thing is that the row was processed successfully

    @patch("yourbench.pipeline.question_rewriting.custom_load_dataset")
    @patch("yourbench.pipeline.question_rewriting.run_inference")
    @patch("yourbench.pipeline.question_rewriting.custom_save_dataset")
    def test_graceful_handling_missing_subset(self, mock_save, mock_inference, mock_load):
        """Test that missing subset is handled gracefully without crashing.

        This test verifies the fix where multi_hop_questions subset might not exist
        in some datasets, and the pipeline should skip gracefully instead of crashing.
        """
        # Mock custom_load_dataset to raise exception for missing subset
        mock_load.side_effect = FileNotFoundError("Subset 'multi_hop_questions' not found in dataset")
        mock_config = Mock()

        # Call _process_question_type - should NOT raise exception
        try:
            _process_question_type(
                config=mock_config,
                question_type="multi-hop",
                load_subset="multi_hop_questions",
                save_subset="multi_hop_questions_rewritten",
                system_prompt="System",
                user_prompt_template="Template",
                additional_instructions="Instructions",
            )
            # If we get here, it means the function handled the missing subset gracefully
            success = True
        except Exception as e:
            # Should not reach here
            success = False
            self.fail(f"Function raised exception for missing subset: {e}")

        # Verify we successfully handled the missing subset
        self.assertTrue(success)

        # Verify custom_load_dataset was called
        mock_load.assert_called_once()

        # Verify inference was NOT called (since subset is missing)
        mock_inference.assert_not_called()

        # Verify save was NOT called (since no data to save)
        mock_save.assert_not_called()

    @patch("yourbench.pipeline.question_rewriting.custom_load_dataset")
    def test_storage_failures_propagate(self, mock_load):
        """Test that non-missing-subset exceptions are caught by outer try-except.

        Note: The function has an outer try-except that catches all exceptions
        and logs them, so even non-'not found' exceptions won't be raised.
        This test verifies that the inner try-except only catches 'not found' errors,
        while the outer try-except handles everything else.
        """
        # Mock custom_load_dataset to raise a different exception
        mock_load.side_effect = Exception("Connection error")
        mock_config = Mock()

        with self.assertRaisesRegex(Exception, "Connection error"):
            _process_question_type(
                config=mock_config,
                question_type="multi-hop",
                load_subset="multi_hop_questions",
                save_subset="multi_hop_questions_rewritten",
                system_prompt="System",
                user_prompt_template="Template",
                additional_instructions="Instructions",
            )

        # Verify custom_load_dataset was called
        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
