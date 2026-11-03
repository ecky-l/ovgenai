#!/bin/bash
#
# Evaluating $(./setupenv.sh) in the current shell exports the three
# variables ovgenai's CMake and the test harness expect:
#   OpenVINO_DIR        = <root of openvino install>/runtime/cmake
#   OpenVINOGenAI_DIR   = <root of openvino.genai install>/runtime/cmake
#   LD_LIBRARY_PATH     = the runtime lib dirs (prepended, not overwriting)
#
# The defaults point at the distro install paths; override by pre-setting
# OPENVINO_DIR / OPENVINO_GENAI_DIR to the parent of your `runtime/` dir.

CWD=$(dirname $(realpath $0))
OV_GENAI_LATEST=$(find ${CWD}/openvinotoolkit -maxdepth 1 -type d | grep openvino_genai | sort | tail -n 1)

: "${OPENVINO_DIR:=${OV_GENAI_LATEST}}"
: "${OPENVINO_GENAI_DIR:=${OV_GENAI_LATEST}}"
echo "export OpenVINO_DIR=\"${OPENVINO_DIR}/runtime/cmake\""
echo "export OpenVINOGenAI_DIR=\"${OPENVINO_GENAI_DIR}/runtime/cmake\""
echo "export LD_LIBRARY_PATH=\"${OPENVINO_GENAI_DIR}/runtime/lib/intel64:${OPENVINO_GENAI_DIR}/runtime/3rdparty/tbb/lib:\$LD_LIBRARY_PATH\""
echo "export PYTHONPATH=${OPENVINO_GENAI_DIR}/python"
