#!/bin/bash

set -eo pipefail

CWD=$(dirname -- $0)
cd ${CWD}
curl -sLO https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/latest.txt
URL=$(grep rhel9 latest.txt | sed -E 's/\w+://')
echo "Fetching openvino.genai from ${URL} ..."
curl -sLO $URL
ARCHIVE=$(basename ${URL})
echo "Extracting ${ARCHIVE} ..."
tar -xzf ${ARCHIVE}
cd -
