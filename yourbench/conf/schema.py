"""
Pydantic config schemas with defaults for YourBench.

Pydantic models define the expected config structure and provide
sensible defaults with built-in validation.
"""

from typing import Any

from pydantic import Field, BaseModel, model_validator


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


class HFConfig(BaseModel):
    """HuggingFace dataset configuration."""

    hf_dataset_name: str = ""
    hf_organization: str = ""
    hf_token: str = Field(default="", repr=False)
    private: bool = False
    concat_if_exist: bool = False
    local_dataset_dir: str = "data/saved_dataset"
    local_saving: bool = True
    upload_card: bool = True
    export_jsonl: bool = False
    jsonl_export_dir: str = "data/jsonl_export"
    push_to_hub: bool = True

    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    """Model configuration."""

    model_name: str = ""
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    max_concurrent_requests: int = 128
    encoding_name: str = "cl100k_base"
    provider: str | None = None
    bill_to: str | None = None
    extra_parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_concurrency(self) -> "ModelConfig":
        if self.max_concurrent_requests < 1:
            raise ConfigValidationError(f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}")
        return self


class ChunkSamplingConfig(BaseModel):
    """Chunk sampling configuration."""

    enable: bool = False
    num_samples: int = 100
    strategy: str = "random"
    random_seed: int = 42

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_samples(self) -> "ChunkSamplingConfig":
        if self.num_samples < 1:
            raise ConfigValidationError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.strategy not in {"random", "first"}:
            raise ConfigValidationError("strategy must be random or first")
        return self


class IngestionConfig(BaseModel):
    """Ingestion stage configuration."""

    run: bool = False
    source_documents_dir: str = "data/raw"
    output_dir: str = "data/processed"
    upload_to_hub: bool = True
    llm_ingestion: bool = False
    pdf_dpi: int = 300
    pdf_llm_prompt: str = ""
    supported_file_extensions: list[str] = Field(default_factory=lambda: [".md", ".txt", ".pdf"])

    model_config = {"extra": "forbid"}


class SummarizationConfig(BaseModel):
    """Summarization stage configuration."""

    run: bool = False
    max_tokens: int = 32768
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    summarization_user_prompt: str = ""
    combine_summaries_user_prompt: str = ""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_tokens(self) -> "SummarizationConfig":
        if self.max_tokens <= 0:
            raise ConfigValidationError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.token_overlap < 0:
            raise ConfigValidationError(f"token_overlap must be >= 0, got {self.token_overlap}")
        if self.token_overlap >= self.max_tokens:
            raise ConfigValidationError(
                f"token_overlap ({self.token_overlap}) must be < max_tokens ({self.max_tokens})"
            )
        return self


class ChunkingConfig(BaseModel):
    """Chunking stage configuration."""

    input_subset: str = "summarized"

    run: bool = False
    l_max_tokens: int = 8192
    token_overlap: int = 512
    encoding_name: str = "cl100k_base"
    h_min: int = 2
    h_max: int = 5
    num_multihops_factor: int = 1

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_chunking(self) -> "ChunkingConfig":
        if self.l_max_tokens <= 0:
            raise ConfigValidationError(f"l_max_tokens must be > 0, got {self.l_max_tokens}")
        if self.token_overlap < 0:
            raise ConfigValidationError(f"token_overlap must be >= 0, got {self.token_overlap}")
        if self.token_overlap >= self.l_max_tokens:
            raise ConfigValidationError("token_overlap must be < l_max_tokens")
        if self.h_min < 1:
            raise ConfigValidationError(f"h_min must be >= 1, got {self.h_min}")
        if self.h_max < self.h_min:
            raise ConfigValidationError(f"h_max ({self.h_max}) must be >= h_min ({self.h_min})")
        if self.num_multihops_factor < 1:
            raise ConfigValidationError(f"num_multihops_factor must be >= 1, got {self.num_multihops_factor}")
        return self


class SingleShotConfig(BaseModel):
    """Single-shot question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    single_hop_system_prompt: str = ""
    single_hop_system_prompt_multi: str = ""
    single_hop_user_prompt: str = ""
    chunk_sampling: ChunkSamplingConfig = Field(default_factory=ChunkSamplingConfig)

    question_schema: str | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_mode(self) -> "SingleShotConfig":
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )
        self.question_mode = mode or "open-ended"
        return self


class MultiHopConfig(BaseModel):
    """Multi-hop question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    multi_hop_system_prompt: str = ""
    multi_hop_system_prompt_multi: str = ""
    multi_hop_user_prompt: str = ""

    question_schema: str | None = None
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_mode(self) -> "MultiHopConfig":
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )
        self.question_mode = mode or "open-ended"
        return self


class CrossDocConfig(BaseModel):
    """Cross-document question generation configuration."""

    run: bool = False
    question_mode: str = "open-ended"
    additional_instructions: str = ""
    multi_hop_system_prompt: str = ""
    multi_hop_system_prompt_multi: str = ""
    multi_hop_user_prompt: str = ""
    max_combinations: int = 100
    chunks_per_document: int = 1
    num_docs_per_combination: list[int] = Field(default_factory=lambda: [2, 5])
    random_seed: int = 42

    model_config = {"extra": "forbid"}

    question_schema: str | None = None

    @model_validator(mode="after")
    def validate_cross_doc(self) -> "CrossDocConfig":
        valid_modes = {"open-ended", "multi-choice", ""}
        mode = self.question_mode.strip().lower() if self.question_mode else ""
        if mode and mode not in valid_modes:
            raise ConfigValidationError(
                f"question_mode must be 'open-ended' or 'multi-choice', got '{self.question_mode}'"
            )
        self.question_mode = mode or "open-ended"
        if self.max_combinations < 1:
            raise ConfigValidationError(f"max_combinations must be >= 1, got {self.max_combinations}")
        if self.chunks_per_document < 1:
            raise ConfigValidationError(f"chunks_per_document must be >= 1, got {self.chunks_per_document}")
        if not isinstance(self.num_docs_per_combination, list) or len(self.num_docs_per_combination) != 2:
            raise ConfigValidationError(
                f"num_docs_per_combination must be a list of 2 elements, got {self.num_docs_per_combination}"
            )
        min_docs, max_docs = self.num_docs_per_combination
        if min_docs < 2:
            raise ConfigValidationError(f"num_docs_per_combination[0] must be >= 2, got {min_docs}")
        if max_docs < min_docs:
            raise ConfigValidationError(f"num_docs_per_combination[1] ({max_docs}) must be >= [0] ({min_docs})")
        return self


class QuestionRewritingConfig(BaseModel):
    """Question rewriting configuration."""

    run: bool = False
    question_rewriting_system_prompt: str = ""
    question_rewriting_user_prompt: str = ""
    additional_instructions: str = ""

    model_config = {"extra": "forbid"}


class LightevalConfig(BaseModel):
    """Lighteval preparation configuration."""

    run: bool = False
    single_hop_subset: str = "single_hop_questions"
    multi_hop_subset: str = "multi_hop_questions"
    cross_doc_subset: str = "cross_document_questions"
    chunked_subset: str = "chunked"
    summarized_subset: str = "summarized"
    output_subset: str = "prepared_lighteval"

    model_config = {"extra": "forbid"}


class CitationFilteringConfig(BaseModel):
    """Citation score filtering configuration."""

    run: bool = False
    subset: str = "prepared_lighteval"
    alpha: float = 0.7
    beta: float = 0.3

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_coefficients(self) -> "CitationFilteringConfig":
        if not (0.0 <= self.alpha <= 1.0):
            raise ConfigValidationError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 <= self.beta <= 1.0):
            raise ConfigValidationError(f"beta must be in [0, 1], got {self.beta}")
        return self


class PipelineConfig(BaseModel):
    """Pipeline configuration with all stages."""

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    single_hop_question_generation: SingleShotConfig = Field(default_factory=SingleShotConfig)
    multi_hop_question_generation: MultiHopConfig = Field(default_factory=MultiHopConfig)
    cross_document_question_generation: CrossDocConfig = Field(default_factory=CrossDocConfig)
    question_rewriting: QuestionRewritingConfig = Field(default_factory=QuestionRewritingConfig)
    prepare_lighteval: LightevalConfig = Field(default_factory=LightevalConfig)
    citation_score_filtering: CitationFilteringConfig = Field(default_factory=CitationFilteringConfig)

    model_config = {"extra": "forbid"}


class YourbenchConfig(BaseModel):
    """Root configuration schema."""

    hf_configuration: HFConfig = Field(default_factory=HFConfig)
    model_list: list[ModelConfig] = Field(default_factory=list)
    model_roles: dict[str, list[str]] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    debug: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_model_roles(self) -> "YourbenchConfig":
        names = [model.model_name for model in self.model_list]
        if any(not name.strip() for name in names):
            raise ConfigValidationError("model_name must not be empty")
        if len(names) != len(set(names)):
            raise ConfigValidationError("model names must be unique; duplicate model_name")
        for stage, assigned in self.model_roles.items():
            unknown = set(assigned) - set(names)
            if unknown:
                raise ConfigValidationError(f"Unknown models assigned to {stage}: {sorted(unknown)}")
        return self
