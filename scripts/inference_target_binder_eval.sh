#!/usr/bin/env bash
set -euo pipefail

python -W ignore multiflow/experiments/inference_target_binder_eval.py -cn inference_target_binder_eval
