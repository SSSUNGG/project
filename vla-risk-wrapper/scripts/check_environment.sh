#!/usr/bin/env bash
# Day 0 environment check script (spec §14)
# Run: bash scripts/check_environment.sh

PASS="✓"
FAIL="✗"
ALL_OK=true

echo "=== VLA Risk Wrapper — Day 0 Environment Check ==="
echo ""

# 1. GPU check
echo "1. GPU"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "   $PASS GPU found: $GPU_NAME"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
        echo "      $line"
    done
else
    echo "   $FAIL nvidia-smi not found (no GPU or driver missing)"
    ALL_OK=false
fi

echo ""

# 2. Python version
echo "2. Python version"
PY_VERSION=$(python3 --version 2>&1 || python --version 2>&1)
MAJOR=$(python3 -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo 0)
MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo 0)
if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
    echo "   $PASS $PY_VERSION (>= 3.10 required)"
else
    echo "   $FAIL $PY_VERSION (need >= 3.10)"
    ALL_OK=false
fi

echo ""

# 3. ManiSkill installed
echo "3. ManiSkill"
if python3 -c "import mani_skill; print('  version:', mani_skill.__version__)" 2>/dev/null; then
    echo "   $PASS mani_skill importable"
else
    echo "   $FAIL mani_skill not installed (pip install mani-skill)"
    ALL_OK=false
fi

echo ""

# 4. Drive mount
echo "4. Google Drive mount"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/vla-risk-wrapper}"
if [ -d "$DRIVE_ROOT" ]; then
    echo "   $PASS DRIVE_ROOT exists: $DRIVE_ROOT"
else
    echo "   $FAIL DRIVE_ROOT not found: $DRIVE_ROOT"
    echo "      (Run: from google.colab import drive; drive.mount('/content/drive'))"
    echo "      (Then: mkdir -p $DRIVE_ROOT/{data,checkpoints,results,figures})"
    ALL_OK=false
fi

echo ""

# 5. HuggingFace token
echo "5. Hugging Face token"
if [ -n "$HF_TOKEN" ]; then
    echo "   $PASS HF_TOKEN is set"
elif [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "   $PASS HUGGING_FACE_HUB_TOKEN is set"
else
    echo "   $FAIL HF_TOKEN not set"
    echo "      (export HF_TOKEN=hf_...)"
    ALL_OK=false
fi

echo ""
echo "==="
if $ALL_OK; then
    echo "All checks passed — Day 0 ready!"
else
    echo "Some checks failed — see above."
fi
