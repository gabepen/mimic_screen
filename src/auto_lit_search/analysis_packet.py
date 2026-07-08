from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GradedPaper(BaseModel):
    paper_id: str
    file_name: str
    paper_role: Optional[str] = None
    grading_schema_version: int = 2
    relevance_grade: float  # legacy 0..1 sort key; derived from weighted axis totals
    relevance_sort: int = 0  # primary-axis weighted total (readable integer rank key)
    paper_grade: str = ""  # e.g. 12/26 across all scored criteria
    primary_grade: str = ""  # e.g. 4/12 on primary relevance axis
    criterion_scores: Dict[str, Any] = Field(default_factory=dict)
    axis_totals: Dict[str, Any] = Field(default_factory=dict)
    rubric_dimension_scores: Dict[str, float] = Field(default_factory=dict)
    rubric_axis_rationales: Dict[str, str] = Field(default_factory=dict)
    mention_type: Optional[str] = None
    infection_naive: Optional[bool] = None
    no_meaningful_mention: bool = False
    claim_summary: str = ""
    rationale: str = ""
    rubric_tags: Dict[str, str] = Field(default_factory=dict)
    model_output: Optional[str] = None
    notes: Optional[str] = None


class RunAlignmentRequest(BaseModel):
    alignment_id: str
    papers_dir: str
    query: str
    target_id: str
    constraints: Optional[Constraints] = None
    instructions: str
    output_root: str
    gene_context: Optional[Dict[str, Any]] = None


class RunAlignmentGradedRequest(RunAlignmentRequest):
    graded_papers: List[GradedPaper]
    grading_meta: Dict[str, Any]


class RunAlignmentResponse(BaseModel):
    status: str
    alignment_id: str
    results_path: str


class GradeAlignmentRequest(RunAlignmentRequest):
    host_rubric_path: str
    microbe_rubric_path: str
    synthesis_host: Optional[str] = None
    synthesis_port: int = 9000

