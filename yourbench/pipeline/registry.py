"""One catalogue of executable stages and their artifact contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    title: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    requires_model: bool = False
    module: str = ""


STAGES = {
    stage.name: stage
    for stage in (
        Stage("ingestion", "Document Ingestion", outputs=("ingested",)),
        Stage("summarization", "Summarization", ("ingested",), ("summarized",), True),
        Stage("chunking", "Chunking", ("summarized",), ("chunked",)),
        Stage(
            "single_hop_question_generation",
            "Single-Hop Questions",
            ("chunked",),
            ("single_hop_questions",),
            True,
            "question_generation.single_hop",
        ),
        Stage(
            "multi_hop_question_generation",
            "Multi-Hop Questions",
            ("chunked",),
            ("multi_hop_questions",),
            True,
            "question_generation.multi_hop",
        ),
        Stage(
            "cross_document_question_generation",
            "Cross-Document Questions",
            ("chunked",),
            ("cross_document_questions",),
            True,
            "question_generation.cross_document",
        ),
        Stage("question_rewriting", "Question Rewriting", requires_model=True),
        Stage("prepare_lighteval", "LightEval Preparation", ("chunked",), ("prepared_lighteval",)),
        Stage("citation_score_filtering", "Citation Scoring", ("prepared_lighteval",), ("prepared_lighteval",)),
    )
}

QUESTION_SUBSETS = {
    "single_hop_question_generation": "single_hop_questions",
    "multi_hop_question_generation": "multi_hop_questions",
    "cross_document_question_generation": "cross_document_questions",
}


def artifacts_for_stage(stage, config):
    """Resolve configurable artifact names without guessing at available data."""
    spec = STAGES[stage]
    cfg = getattr(config.pipeline, stage)
    if stage == "chunking":
        return (cfg.input_subset,), spec.outputs
    if stage == "prepare_lighteval":
        return (cfg.chunked_subset,), (cfg.output_subset,)
    if stage == "citation_score_filtering":
        return (cfg.subset,), (cfg.subset,)
    if stage == "question_rewriting":
        return (), tuple(
            f"{subset}_rewritten" for name, subset in QUESTION_SUBSETS.items() if getattr(config.pipeline, name).run
        )
    return spec.inputs, spec.outputs
