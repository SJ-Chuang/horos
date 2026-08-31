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

__all__ = [
    "AnnotationProgress",
    "AnnotationSetView",
    "AssistResult",
    "Capability",
    "ClaimResult",
    "DoctorReport",
    "ImportSummary",
    "JobStatus",
    "PlatformCapabilities",
    "PromptSpec",
    "QueueItem",
    "add_category",
    "annotation_progress",
    "assist_image",
    "cancel_job",
    "claim_image",
    "convert_dataset",
    "create_project",
    "dataset_stats",
    "delete_category",
    "doctor_report",
    "ensure_supported",
    "export_dataset",
    "get_annotations",
    "get_capability",
    "image_file_path",
    "image_queue",
    "import_dataset",
    "import_zip",
    "job_status",
    "list_capabilities",
    "list_images",
    "list_models",
    "open_project",
    "pending_summary",
    "platform_capabilities",
    "release_claim",
    "resplit",
    "review_pending",
    "save_annotations",
    "start_autolabel",
    "update_category",
    "validate_project",
]
