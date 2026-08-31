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
from horos.api.labels import add_category, delete_category, update_category
from horos.api.manifest import Capability, get_capability, list_capabilities
from horos.api.project import create_project, open_project
from horos.api.system import (
    PlatformCapabilities,
    ensure_supported,
    list_models,
    platform_capabilities,
)

__all__ = [
    "AnnotationProgress",
    "AnnotationSetView",
    "Capability",
    "ClaimResult",
    "ImportSummary",
    "PlatformCapabilities",
    "QueueItem",
    "add_category",
    "annotation_progress",
    "claim_image",
    "convert_dataset",
    "create_project",
    "dataset_stats",
    "delete_category",
    "ensure_supported",
    "export_dataset",
    "get_annotations",
    "get_capability",
    "image_file_path",
    "image_queue",
    "import_dataset",
    "import_zip",
    "list_capabilities",
    "list_images",
    "list_models",
    "open_project",
    "platform_capabilities",
    "release_claim",
    "resplit",
    "save_annotations",
    "update_category",
    "validate_project",
]
