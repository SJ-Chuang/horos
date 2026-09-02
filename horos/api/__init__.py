"""horos public Python API (R2: the business-logic layer).

Everything the Web API and CLI expose routes through functions defined in this
package; each public function is registered in the capability manifest (E9-T1).
"""

from horos.api.annotate import (
    AnnotationProgress,
    AnnotationSetView,
    ClaimResult,
    QueueItem,
    annotation_progress,
    claim_image,
    get_annotations,
    image_file_path,
    image_queue,
    release_claim,
    save_annotations,
)
from horos.api.autolabel import (
    AssistResult,
    PromptSpec,
    assist_image,
    pending_summary,
    review_pending,
    start_autolabel,
)
from horos.api.dataset import (
    ImportSummary,
    convert_dataset,
    dataset_stats,
    export_dataset,
    import_dataset,
    import_zip,
    list_images,
    resplit,
    validate_project,
)
from horos.api.evaluate import (
    ClassEval,
    EvalReport,
    get_eval_report,
    infer_image,
    start_evaluation,
)
from horos.api.hparams import DerivedValue, HyperparameterPlan
from horos.api.jobs import JobStatus, cancel_job, job_status
from horos.api.labels import add_category, delete_category, update_category
from horos.api.manifest import Capability, get_capability, list_capabilities
from horos.api.project import create_project, open_project
from horos.api.system import (
    DoctorReport,
    PlatformCapabilities,
    doctor_report,
    ensure_supported,
    list_models,
    platform_capabilities,
)
from horos.api.train import (
    RunRecord,
    TrainRunConfig,
    TrainStatus,
    delete_run,
    derive_hyperparameters,
    list_runs,
    start_training,
    stop_training,
    training_status,
    update_queued_run,
)
from horos.api.verdict import Finding, RunVerdict, run_verdict

__all__ = [
    "AnnotationProgress",
    "AnnotationSetView",
    "AssistResult",
    "Capability",
    "ClaimResult",
    "ClassEval",
    "DerivedValue",
    "DoctorReport",
    "EvalReport",
    "Finding",
    "HyperparameterPlan",
    "ImportSummary",
    "JobStatus",
    "PlatformCapabilities",
    "PromptSpec",
    "QueueItem",
    "RunRecord",
    "RunVerdict",
    "TrainRunConfig",
    "TrainStatus",
    "add_category",
    "annotation_progress",
    "assist_image",
    "cancel_job",
    "claim_image",
    "convert_dataset",
    "create_project",
    "dataset_stats",
    "delete_category",
    "delete_run",
    "derive_hyperparameters",
    "doctor_report",
    "ensure_supported",
    "export_dataset",
    "get_annotations",
    "get_capability",
    "get_eval_report",
    "image_file_path",
    "image_queue",
    "import_dataset",
    "import_zip",
    "infer_image",
    "job_status",
    "list_capabilities",
    "list_images",
    "list_models",
    "list_runs",
    "open_project",
    "pending_summary",
    "platform_capabilities",
    "release_claim",
    "resplit",
    "review_pending",
    "run_verdict",
    "save_annotations",
    "start_autolabel",
    "start_evaluation",
    "start_training",
    "stop_training",
    "training_status",
    "update_category",
    "update_queued_run",
    "validate_project",
]
