"""Explicit configuration and provenance support for trajectory workflows.

These modules do not alter the production estimator.  They make the inputs to
the historical trajectory-to-effect route inspectable before a run begins.
"""

from .schema import (
    AggregationConfig,
    DownstreamExportConfig,
    UncertaintyConfig,
    WorkflowConfig,
    WorkflowConfigError,
    load_workflow_config,
)
from .executor import WorkflowExecutionError, execute_workflow
from .downstream import DownstreamExportError, export_downstream_layout
from .summary import WorkflowSummaryError, summarize_to_downstream_dataset
from .validation import ValidationResult, validate_workflow_config

__all__ = [
    "WorkflowConfig",
    "UncertaintyConfig",
    "AggregationConfig",
    "DownstreamExportConfig",
    "WorkflowConfigError",
    "WorkflowExecutionError",
    "execute_workflow",
    "DownstreamExportError",
    "export_downstream_layout",
    "WorkflowSummaryError",
    "summarize_to_downstream_dataset",
    "ValidationResult",
    "load_workflow_config",
    "validate_workflow_config",
]
