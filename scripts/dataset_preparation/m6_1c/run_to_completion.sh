#!/usr/bin/env bash
# Resume with one network reader, then audit, package, and refresh the report.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=/home/namng/anaconda3/envs/filmgrasp39/bin/python
RUN=runs/dataset_preparation/m6_1c

"$PY" scripts/dataset_preparation/m6_1c/prepare_expanded.py --workers 1
extract_status=$?
if [ "$extract_status" -eq 0 ]; then
    printf '%s\n' 'none recorded' > "$RUN/blocker.txt"
else
    printf 'Source extraction exited with status %s; inspect extraction.log and rerun this script to resume.\n' "$extract_status" > "$RUN/blocker.txt"
fi
"$PY" scripts/dataset_preparation/m6_1c/audit_and_package.py --package
audit_status=$?
"$PY" scripts/dataset_preparation/m6_1c/write_report.py
report_status=$?
if [ "$extract_status" -ne 0 ] || [ "$audit_status" -ne 0 ] || [ "$report_status" -ne 0 ]; then
    exit 1
fi
