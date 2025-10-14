# N-gram Analysis Tool

This tool identifies important n-gram patterns between two kinds of model responses by analyzing frequency distributions, statistical significance, and pattern characteristics.

## Features

- **Dynamic N-gram Analysis**: Automatically encourages longer n-grams when they provide more distinguishing information
- **Multiple Statistical Measures**: Mutual information, chi-square tests, log odds ratios, frequency ratios, and information gain
- **Variable-Length N-grams**: Analyze n-grams of varying lengths (1-10+ characters) in a single analysis
- **Length Bonus System**: Configurable bonus factor that favors longer n-grams when they're statistically significant
- **Comparative Analysis**: Identify patterns that distinguish between two datasets
- **Robust Statistics**: Handles missing dependencies gracefully
- **Command-line Interface**: Easy to use with customizable parameters

## Installation

### Basic Installation (no external dependencies)
The tool works with Python's standard library for basic functionality:
```bash
python3 ngram/analysis.py file1.json file2.json
```

### Full Installation (with statistical tests)
For advanced statistical tests, install the optional dependencies:
```bash
pip install -r ngram/requirements.txt
```

## Usage

### Basic Usage (Dynamic Mode - Recommended)
```bash
python3 ngram/analysis.py data/sequence/reasoning.json data/sequence/non_reasoning.json --dynamic
```

### Advanced Dynamic Usage
```bash
python3 ngram/analysis.py file1.json file2.json \
    --dynamic \
    --min-length 2 \
    --max-length 8 \
    -k 15 \
    --min-count 3 \
    --length-bonus 0.1 \
    --labels "Reasoning Models" "Non-Reasoning Models"
```

### Traditional Fixed-Length Usage
```bash
python3 ngram/analysis.py file1.json file2.json \
    -n 2 3 4 \
    -k 15 \
    --min-count 3 \
    --labels "Reasoning Models" "Non-Reasoning Models"
```

### Parameters

- `file1`, `file2`: JSON files containing arrays of sequences
- `--dynamic`: Use dynamic n-gram analysis (recommended)
- `--min-length`: Minimum n-gram length for dynamic analysis (default: 1)
- `--max-length`: Maximum n-gram length for dynamic analysis (default: 10)
- `--length-bonus`: Length bonus factor for encouraging longer n-grams (default: 0.1)
- `-n, --ngram-sizes`: N-gram sizes to analyze for fixed-length mode (default: 2 3 4)
- `-k, --top-k`: Number of top patterns to show (default: 20)
- `--min-count`: Minimum count threshold for n-grams (default: 3)
- `--labels`: Custom labels for the datasets (default: Dataset1 Dataset2)

## Output Interpretation

The tool provides several types of analysis:

### Dynamic Mode Analysis

#### 1. Information Gain
- Measures how much information an n-gram provides for distinguishing between datasets
- Based on entropy reduction when considering n-gram presence/absence
- Higher values indicate more informative patterns

#### 2. Information Gain with Length Bonus
- Same as information gain but with a bonus factor that encourages longer n-grams
- Longer n-grams get higher scores when they're statistically significant
- Helps identify complex patterns that might be missed by shorter n-grams

#### 3. Mutual Information (Variable Length)
- Measures statistical dependence between n-gram presence and dataset membership
- Works across all n-gram lengths simultaneously
- Higher values indicate stronger association

#### 4. Mutual Information with Length Bonus
- Same as mutual information but with length bonus applied
- Encourages discovery of longer, more specific patterns

### Traditional Mode Analysis

#### 1. Mutual Information
- Measures how much information one dataset provides about the presence of an n-gram
- Higher values indicate stronger association with one dataset

#### 2. Chi-Square Test
- Tests statistical significance of differences between datasets
- Only available with scipy installed
- Lower p-values indicate more significant differences

#### 3. Log Odds Ratio
- Positive values favor Dataset1, negative values favor Dataset2
- Magnitude indicates strength of association

#### 4. Frequency Ratios
- Direct comparison of relative frequencies between datasets
- Higher ratios indicate stronger preference for one dataset

### Summary Statistics
- Total and unique n-gram counts
- Length distribution (in dynamic mode)
- Overlap analysis between datasets

## Example Output

### Dynamic Mode Output
```
======================================================================
DYNAMIC N-GRAM ANALYSIS RESULTS (lengths 2-8)
======================================================================
Dataset 1 (Reasoning): 36 sequences
Dataset 2 (Non-Reasoning): 45 sequences
Length range: 2-8
Minimum count threshold: 2
Length bonus factor: 0.1

------------------------------------------------------------
TOP PATTERNS BY INFORMATION GAIN WITH LENGTH BONUS
------------------------------------------------------------
 1. 'AV' (length: 2, score: 0.7986)
    Reasoning:   99 | Non-Reasoning:    3
 2. 'VA' (length: 2, score: 0.7595)
    Reasoning:  127 | Non-Reasoning:    6
 3. 'VAV' (length: 3, score: 0.6724)
    Reasoning:   47 | Non-Reasoning:    0
```

### Traditional Mode Output
```
============================================================
N-GRAM ANALYSIS RESULTS (n=2)
============================================================
Dataset 1 (Reasoning): 36 sequences
Dataset 2 (Non-Reasoning): 45 sequences

--------------------------------------------------
TOP PATTERNS BY MUTUAL INFORMATION
--------------------------------------------------
 1. 'AV' (score: 0.7196)
    Reasoning:   99 | Non-Reasoning:    3
 2. 'VA' (score: 0.6843)
    Reasoning:  127 | Non-Reasoning:    6
```

## Data Format

Input files should be JSON arrays of strings:
```json
[
    "RMPRNEPIMPIMNEPIMIMNEPIPIVIVPIVMEPIPIEMEVMINMINMEPIVMNVMVRVPIPIVPNPIVREPIPIVPIVNVIVNVRNAMPENEMPNPININPIVINEMNPIMPINVNVNVNAVIVIVEVNAENIRINPINVAPIEVAVA",
    "RMENVNVPNIVPIPIVNIAVIVNAEIVNIAMPININVANIA",
    "RMPRNEPNMNVMPENMNINIPIPIVIVIPMENVINIVIAVRPIPNPIVIPIVAININIA"
]
```

## Why Use Dynamic Mode?

The dynamic n-gram analysis offers several advantages over traditional fixed-length approaches:

1. **Automatic Length Selection**: Instead of manually choosing n-gram sizes, the tool automatically finds the most informative patterns regardless of length
2. **Longer Pattern Discovery**: The length bonus system encourages discovery of longer, more specific patterns that might be missed by shorter n-grams
3. **Comprehensive Coverage**: Analyzes all possible n-gram lengths in a single run, providing complete pattern coverage
4. **Statistical Rigor**: Uses information gain and mutual information to ensure discovered patterns are statistically meaningful
5. **Efficiency**: Single analysis replaces multiple fixed-length analyses

## Applications

This tool is particularly useful for:
- Comparing reasoning vs non-reasoning model outputs
- Identifying characteristic patterns in different model behaviors
- Analyzing linguistic or sequential differences between datasets
- Feature extraction for machine learning models
- Discovering complex multi-character patterns in encoded sequences
