#!/usr/bin/env bash
# Fetch the CorrDiff-Taiwan pretrained package and reference code (GPU workstation).
#
#   bash scripts/setup/get_corrdiff.sh [checkpoint_dir]
#
# Package: NGC nvidia/modulus/corrdiff_inference_package v1 (~684 MB zip).
#   checkpoints (diffusion.mdlus, regression.mdlus): Apache-2.0
#   bundled sample data: CC BY-NC-ND 4.0 (research use only, no redistribution)
# Earth2Studio downloads the same package automatically
# (CorrDiffTaiwan.load_default_package()); this script is for working with the
# original PhysicsNeMo (ex-Modulus) generate.py or for offline machines.
set -euo pipefail

DEST="${1:-checkpoints/corrdiff_taiwan}"
mkdir -p "$DEST" third_party

ZIP="$DEST/corrdiff_inference_package_1.zip"
if [ ! -f "$ZIP" ]; then
  wget --content-disposition \
    "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/corrdiff_inference_package/versions/1/zip" \
    -O "$ZIP"
fi
unzip -n -q "$ZIP" -d "$DEST"
find "$DEST" -name "*.mdlus" -print

# Reference implementation at the commit the NGC package was released with
# (the repository was renamed from NVIDIA/modulus to NVIDIA/physicsnemo).
if [ ! -d third_party/physicsnemo ]; then
  git clone https://github.com/NVIDIA/physicsnemo.git third_party/physicsnemo
  git -C third_party/physicsnemo checkout ee6b3ac0d4c32bca1c81f401d2bfd461887eb9a3
fi

cat <<'EOF'

Next steps
  pip install "earth2studio[corrdiff]"
  python scripts/setup/corrdiff_smoke_test.py --time 2021-09-11T21:00 --samples 4
EOF
