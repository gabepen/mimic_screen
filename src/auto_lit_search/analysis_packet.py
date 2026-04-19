from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GradedPaper(BaseModel):
    paper_id: str
    file_name: str
    paper_role: Optional[str] = None
    relevance_grade: float  # computed server-side from rubric_dimension_scores
    rubric_dimension_scores: Dict[str, float]
    rubric_axis_rationales: Dict[str, str] = Field(default_factory=dict)
    rationale: str = ""  # optional brief cross-axis summary
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
    synthesis_host: str
    synthesis_port: int = 9000

