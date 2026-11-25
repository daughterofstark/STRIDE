"""Explicit configuration and provenance support for trajectory workflows.

These modules do not alter the production estimator.  They make the inputs to
the historical trajectory-to-effect route inspectable before a run begins.
"""

from .downstream import DownstreamExportError, export_downstream_layout
from .executor import WorkflowExecutionError, execute_workflow
from .schema import (
    AggregationConfig,
    DownstreamExportConfig,
    UncertaintyConfig,
    WorkflowConfig,
    WorkflowConfigError,
    load_workflow_config,
)
from .summary import WorkflowSummaryError, summarize_to_downstream_dataset
from .validation import ValidationResult, validate_workflow_config

__all__ = [
    "AggregationConfig",
    "DownstreamExportConfig",
    "DownstreamExportError",
    "UncertaintyConfig",
    "ValidationResult",
    "WorkflowConfig",
    "WorkflowConfigError",
    "WorkflowExecutionError",
    "WorkflowSummaryError",
    "execute_workflow",
    "export_downstream_layout",
    "load_workflow_config",
    "summarize_to_downstream_dataset",
    "validate_workflow_config",
]
