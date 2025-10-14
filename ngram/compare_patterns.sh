#!/bin/bash

# Simple wrapper script for n-gram analysis
# Usage: ./compare_patterns.sh [file1] [file2] [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analysis.py"

# Default files
DEFAULT_FILE1="$SCRIPT_DIR/../data/sequence/reasoning.json"
DEFAULT_FILE2="$SCRIPT_DIR/../data/sequence/non_reasoning.json"

# Check if first two arguments are files (not options starting with -)
if [ $# -ge 2 ] && [[ "$1" != -* ]] && [[ "$2" != -* ]]; then
    FILE1="$1"
    FILE2="$2"
    shift 2
else
    FILE1="$DEFAULT_FILE1"
    FILE2="$DEFAULT_FILE2"
fi

echo "Comparing n-gram patterns between:"
echo "  File 1: $FILE1"
echo "  File 2: $FILE2"
echo ""

# Run the analysis with remaining arguments
python3 "$ANALYSIS_SCRIPT" "$FILE1" "$FILE2" \
    --labels "Reasoning" "Non-Reasoning" \
    --dynamic \
    --min-length 2 \
    --max-length 8 \
    -k 15 \
    --min-count 2 \
    --length-bonus 0.5 \
    "$@"
