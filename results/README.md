# Inter-Annotator Agreement Analysis Results

This directory contains the results of Cohen's Kappa analysis for inter-annotator agreement between two annotators (gemini-2.5-flash and gpt-5).

## Files

### 1. `kappa_scores.txt`
Quick summary of kappa scores by model and overall, with key statistics and recommendations.

**Key Finding**: Overall Cohen's Kappa = 0.6599 (Substantial Agreement)

### 2. `kappa_summary.md`
Comprehensive analysis document including:
- Detailed methodology
- Results by model and category
- Confusion patterns
- Interpretation and recommendations
- Statistical significance discussion

### 3. `inter_annotator_agreement.txt`
Full detailed output including:
- Confusion matrices for each model
- Per-category precision and recall
- Breakdown of all disagreements

## Quick Summary

| Model | Cohen's Kappa | Annotations | Agreement |
|-------|---------------|-------------|-----------|
| deepseekR1 | **0.7515** | 1,451 | 80.5% |
| Phi4R | **0.7512** | 886 | 78.7% |
| Qwen3_32B | **0.5599** | 2,287 | 64.3% |
| **Overall** | **0.6599** | **4,624** | **72.2%** |

## Category Agreement (Overall)

Categories ranked by recall (how often A1's label was agreed upon by A2):

1. **Implement** - 86.1% recall ✓ Most reliable
2. **Answer** - 81.1% recall ✓ Clear category
3. **Read** - 77.9% recall ✓ Good agreement
4. **Explore** - 72.3% recall ~ Moderate
5. **Verify** - 67.6% recall ~ Moderate
6. **Analyze** - 64.4% recall ~ Challenging
7. **Plan** - 61.1% recall ⚠ Challenging
8. **Monitor** - 53.2% recall ⚠ Most difficult

## Key Insights

### What This Means
- **κ = 0.66 is substantial agreement** for complex cognitive categorization (8 categories, nuanced distinctions)
- **Concrete categories (Implement, Answer, Read) have high agreement** (77-86% recall)
- **Abstract/meta-cognitive categories (Monitor, Plan, Analyze) are more challenging** (53-64% recall)
- **Model differences**: deepseekR1 and Phi4R produce clearer reasoning (κ≈0.75) than Qwen3_32B (κ=0.56)

### Most Common Confusions
1. **Verify ↔ Implement**: Checking involves computation
2. **Analyze ↔ Verify**: Inference vs. validation are similar
3. **Plan ↔ Implement**: Setting up vs. executing
4. **Monitor**: Confused with many categories (meta-cognitive is hard to isolate)

## Recommendations

1. **Acceptable for research**: κ=0.66 is good enough for analysis, but borderline cases should be reviewed
2. **Focus improvement on**: Monitor (53%), Plan (61%), and Qwen3_32B data (56%)
3. **Consider**: Third annotator for adjudication, or consensus coding for training data
4. **Guidelines**: Need clearer operational definitions for Monitor, Plan, and Analyze categories

## How to Reproduce

Run the analysis scripts:
```bash
# Basic kappa scores
python3 temp/kappa.py

# Detailed analysis with confusion matrices
python3 temp/kappa_detailed.py

# Or save to file
python3 temp/kappa_detailed.py > results/inter_annotator_agreement.txt
```

## Statistical Significance

With n=4,624 annotations, these kappa scores are highly statistically significant (p < 0.001). The differences between models are also meaningful, not due to chance.

## Citation

If using this analysis, please cite:
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.

---

*Analysis generated: 2025-10-07*  
*Data: /fs/nexus-scratch/cfan42/reasoning-insider/data/label/*

