#!/bin/bash
set -euo pipefail

cd /opt/physical-ai-system
export PYTHONPATH=/opt/physical-ai-system/src
export AWS_REGION=ap-northeast-3
export S3_BUCKET=physical-ai-development-system-bucket
export LOCAL_DATA_DIR=/opt/physical-ai-system/local_data

/opt/physical-ai-system/.venv/bin/python -m physical_ai_system.cli_monthly_scrape >> /opt/physical-ai-system/local_data/monthly_cron.log 2>&1
