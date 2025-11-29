---
description: Audit implementation status of all specs and generate report
allowed-tools: Bash, Read, Write
---

# Audit Specs

Run the spec audit tool to check implementation status of all specs.

## Variables

RUN_VALIDATION: $1 (optional: "true" to run validation commands)

## Workflow

1. Run the audit tool: `python tools/audit_specs.py --run-validation` (if RUN_VALIDATION is true, otherwise run without flag)
2. Read and display the generated STATUS_REPORT.md
3. Highlight high-priority pending specs

## Report

- Total specs scanned
- Breakdown by status
- High priority items requiring attention
