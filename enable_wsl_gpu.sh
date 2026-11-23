#!/usr/bin/env bash
# Source this after activating the assignment venv and before starting Python.
# TensorFlow 2.21.0 omits the pip-installed cuSOLVER directory from one RUNPATH.

if ! command -v python >/dev/null 2>&1; then
  echo "Activate ~/.venvs/cnn-assignment02 before sourcing this file." >&2
  return 1
fi

_cnn_cusolver_lib="$(python -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("nvidia.cusolver"); assert spec and spec.submodule_search_locations, "nvidia-cusolver-cu12 is missing"; print(pathlib.Path(next(iter(spec.submodule_search_locations))) / "lib")')" || return 1
if [ ! -f "$_cnn_cusolver_lib/libcusolver.so.11" ]; then
  echo "Missing $_cnn_cusolver_lib/libcusolver.so.11" >&2
  unset _cnn_cusolver_lib
  return 1
fi

case ":${LD_LIBRARY_PATH:-}:" in
  *":$_cnn_cusolver_lib:"*) ;;
  *) export LD_LIBRARY_PATH="$_cnn_cusolver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
echo "Enabled TensorFlow cuSOLVER library path for this shell."
unset _cnn_cusolver_lib
