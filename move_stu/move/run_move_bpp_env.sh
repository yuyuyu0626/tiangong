#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/2024233240/miniconda3/envs/move_bpp"
export PATH="${ENV_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_ROOT}/lib:${LD_LIBRARY_PATH:-}"

exec "${ENV_ROOT}/bin/python" "$@"
