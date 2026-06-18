#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/home/ubuntu/anaconda3/envs/bpp"
export PATH="${ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

exec "${ENV_ROOT}/bin/python" "$@"
