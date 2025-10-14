#!/usr/bin/env python3
"""
N-gram Analysis Tool for Comparing Model Response Patterns

This tool identifies important n-gram patterns between two kinds of model responses
by analyzing frequency distributions, statistical significance, and pattern characteristics.
"""

import json
import argparse
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Set

# Optional imports for advanced statistics
try:
    import numpy as np
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some statistical tests will be skipped.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class NGramAnalyzer:
    """Analyzes n-gram patterns between two sets of sequences."""
    
    def __init__(self, sequences1: List[str], sequences2: List[str], label1: str = "Dataset1", label2: str = "Dataset2"):
        self.sequences1 = sequences1
        self.sequences2 = sequences2
        self.label1 = label1
        self.label2 = label2
        self.ngram_counts1 = {}
        self.ngram_counts2 = {}
        self.total_ngrams1 = 0
        self.total_ngrams2 = 0
        self.variable_ngrams = {}  # For variable-length analysis
    
    def extract_ngrams(self, sequence: str, n: int) -> List[str]:
        """Extract n-grams from a sequence."""
        if len(sequence) < n:
            return [sequence] if sequence else []
        return [sequence[i:i+n] for i in range(len(sequence) - n + 1)]
    
    def extract_variable_length_ngrams(self, sequence: str, min_len: int = 1, max_len: int = 10) -> List[Tuple[str, int]]:
        """Extract n-grams of variable lengths from a sequence.
        
        Returns:
            List of tuples (ngram, length) for all n-grams from min_len to max_len
        """
        ngrams = []
        for n in range(min_len, min(max_len + 1, len(sequence) + 1)):
            if len(sequence) >= n:
                for i in range(len(sequence) - n + 1):
                    ngram = sequence[i:i+n]
                    ngrams.append((ngram, n))
        return ngrams
    
    def count_variable_length_ngrams(self, min_len: int = 1, max_len: int = 10) -> None:
        """Count variable-length n-grams in both datasets."""
        self.variable_ngrams = {
            'dataset1': Counter(),
            'dataset2': Counter(),
            'lengths': {}
        }
        
        # Count n-grams in dataset 1
        for seq in self.sequences1:
            ngrams = self.extract_variable_length_ngrams(seq, min_len, max_len)
            for ngram, length in ngrams:
                self.variable_ngrams['dataset1'][ngram] += 1
                if ngram not in self.variable_ngrams['lengths']:
                    self.variable_ngrams['lengths'][ngram] = length
        
        # Count n-grams in dataset 2
        for seq in self.sequences2:
            ngrams = self.extract_variable_length_ngrams(seq, min_len, max_len)
            for ngram, length in ngrams:
                self.variable_ngrams['dataset2'][ngram] += 1
                if ngram not in self.variable_ngrams['lengths']:
                    self.variable_ngrams['lengths'][ngram] = length
    
    def count_ngrams(self, n: int) -> None:
        """Count n-grams in both datasets."""
        self.ngram_counts1[n] = Counter()
        self.ngram_counts2[n] = Counter()
        
        # Count n-grams in dataset 1
        for seq in self.sequences1:
            ngrams = self.extract_ngrams(seq, n)
            self.ngram_counts1[n].update(ngrams)
        
        # Count n-grams in dataset 2
        for seq in self.sequences2:
            ngrams = self.extract_ngrams(seq, n)
            self.ngram_counts2[n].update(ngrams)
        
        self.total_ngrams1 = sum(self.ngram_counts1[n].values())
        self.total_ngrams2 = sum(self.ngram_counts2[n].values())
    
    def calculate_frequencies(self, n: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate relative frequencies for n-grams."""
        freq1 = {ngram: count / self.total_ngrams1 
                for ngram, count in self.ngram_counts1[n].items()}
        freq2 = {ngram: count / self.total_ngrams2 
                for ngram, count in self.ngram_counts2[n].items()}
        return freq1, freq2
    
    def calculate_mutual_information(self, n: int) -> Dict[str, float]:
        """Calculate mutual information for each n-gram."""
        mi_scores = {}
        all_ngrams = set(self.ngram_counts1[n].keys()) | set(self.ngram_counts2[n].keys())
        
        total_sequences = len(self.sequences1) + len(self.sequences2)
        
        for ngram in all_ngrams:
            # Count occurrences
            count_1_with = sum(1 for seq in self.sequences1 if ngram in seq)
            count_1_without = len(self.sequences1) - count_1_with
            count_2_with = sum(1 for seq in self.sequences2 if ngram in seq)
            count_2_without = len(self.sequences2) - count_2_with
            
            # Create contingency table
            contingency = np.array([[count_1_with, count_1_without],
                                  [count_2_with, count_2_without]])
            
            # Calculate mutual information
            if contingency.sum() > 0:
                mi = self._mutual_information_from_contingency(contingency)
                mi_scores[ngram] = mi
            else:
                mi_scores[ngram] = 0.0
        
        return mi_scores
    
    def _mutual_information_from_contingency(self, contingency: np.ndarray) -> float:
        """Calculate mutual information from contingency table."""
        total = contingency.sum()
        if total == 0:
            return 0.0
        
        mi = 0.0
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                if contingency[i, j] > 0:
                    p_xy = contingency[i, j] / total
                    p_x = contingency[i, :].sum() / total
                    p_y = contingency[:, j].sum() / total
                    if p_x > 0 and p_y > 0:
                        mi += p_xy * math.log2(p_xy / (p_x * p_y))
        
        return mi
    
    def calculate_chi_square(self, n: int, min_count: int = 5) -> Dict[str, Tuple[float, float]]:
        """Calculate chi-square statistics for n-grams."""
        chi2_results = {}
        
        if not SCIPY_AVAILABLE:
            return chi2_results
        
        all_ngrams = set(self.ngram_counts1[n].keys()) | set(self.ngram_counts2[n].keys())
        
        for ngram in all_ngrams:
            count1 = self.ngram_counts1[n].get(ngram, 0)
            count2 = self.ngram_counts2[n].get(ngram, 0)
            
            # Skip if counts are too low
            if count1 + count2 < min_count:
                continue
            
            # Create contingency table
            total1 = self.total_ngrams1
            total2 = self.total_ngrams2
            
            observed = np.array([[count1, total1 - count1],
                               [count2, total2 - count2]])
            
            try:
                chi2, p_value, _, _ = chi2_contingency(observed)
                chi2_results[ngram] = (chi2, p_value)
            except ValueError:
                # Handle cases where chi2 test is not applicable
                continue
        
        return chi2_results
    
    def calculate_log_odds_ratio(self, n: int) -> Dict[str, float]:
        """Calculate log odds ratio for each n-gram."""
        log_odds = {}
        all_ngrams = set(self.ngram_counts1[n].keys()) | set(self.ngram_counts2[n].keys())
        
        for ngram in all_ngrams:
            count1 = self.ngram_counts1[n].get(ngram, 0)
            count2 = self.ngram_counts2[n].get(ngram, 0)
            
            # Add smoothing to avoid division by zero
            smoothing = 0.5
            
            # Calculate odds
            odds1 = (count1 + smoothing) / (self.total_ngrams1 - count1 + smoothing)
            odds2 = (count2 + smoothing) / (self.total_ngrams2 - count2 + smoothing)
            
            # Calculate log odds ratio
            log_odds[ngram] = math.log(odds1 / odds2)
        
        return log_odds
    
    def calculate_information_gain(self, min_len: int = 1, max_len: int = 10) -> Dict[str, float]:
        """Calculate information gain for variable-length n-grams.
        
        Information gain measures how much information an n-gram provides
        for distinguishing between the two datasets.
        """
        self.count_variable_length_ngrams(min_len, max_len)
        
        info_gain = {}
        all_ngrams = set(self.variable_ngrams['dataset1'].keys()) | set(self.variable_ngrams['dataset2'].keys())
        
        total_sequences = len(self.sequences1) + len(self.sequences2)
        
        for ngram in all_ngrams:
            # Count sequences containing this n-gram
            count_1_with = sum(1 for seq in self.sequences1 if ngram in seq)
            count_1_without = len(self.sequences1) - count_1_with
            count_2_with = sum(1 for seq in self.sequences2 if ngram in seq)
            count_2_without = len(self.sequences2) - count_2_with
            
            # Calculate entropy before and after
            # Entropy before (overall class distribution)
            p_class1 = len(self.sequences1) / total_sequences
            p_class2 = len(self.sequences2) / total_sequences
            entropy_before = 0
            if p_class1 > 0:
                entropy_before -= p_class1 * math.log2(p_class1)
            if p_class2 > 0:
                entropy_before -= p_class2 * math.log2(p_class2)
            
            # Entropy after (conditional on n-gram presence/absence)
            total_with = count_1_with + count_2_with
            total_without = count_1_without + count_2_without
            
            entropy_after = 0
            
            if total_with > 0:
                p1_with = count_1_with / total_with
                p2_with = count_2_with / total_with
                entropy_with = 0
                if p1_with > 0:
                    entropy_with -= p1_with * math.log2(p1_with)
                if p2_with > 0:
                    entropy_with -= p2_with * math.log2(p2_with)
                entropy_after += (total_with / total_sequences) * entropy_with
            
            if total_without > 0:
                p1_without = count_1_without / total_without
                p2_without = count_2_without / total_without
                entropy_without = 0
                if p1_without > 0:
                    entropy_without -= p1_without * math.log2(p1_without)
                if p2_without > 0:
                    entropy_without -= p2_without * math.log2(p2_without)
                entropy_after += (total_without / total_sequences) * entropy_without
            
            # Information gain = entropy_before - entropy_after
            info_gain[ngram] = entropy_before - entropy_after
        
        return info_gain
    
    def calculate_length_penalty_score(self, ngram: str, base_score: float, length_bonus: float = 0.1) -> float:
        """Calculate a score that encourages longer n-grams.
        
        Args:
            ngram: The n-gram string
            base_score: The base statistical score (e.g., mutual information)
            length_bonus: Bonus factor for length (default: 0.1)
        
        Returns:
            Adjusted score that favors longer n-grams
        """
        length = len(ngram)
        # Apply a logarithmic bonus for length to encourage longer n-grams
        # but not too aggressively
        length_adjustment = 1 + length_bonus * math.log(length + 1)
        return base_score * length_adjustment
    
    def find_dynamic_patterns(self, min_len: int = 1, max_len: int = 10, top_k: int = 20, 
                            min_count: int = 3, length_bonus: float = 0.1) -> Dict[str, List[Tuple[str, float, int]]]:
        """Find the most informative n-grams using dynamic length selection.
        
        This method encourages longer n-grams by applying length bonuses to statistical scores.
        
        Returns:
            Dictionary with different ranking methods, each containing tuples of (ngram, score, length)
        """
        # Calculate information gain for all variable-length n-grams
        info_gain = self.calculate_information_gain(min_len, max_len)
        
        # Filter by minimum count
        filtered_ngrams = set()
        for ngram in info_gain.keys():
            total_count = (self.variable_ngrams['dataset1'].get(ngram, 0) + 
                          self.variable_ngrams['dataset2'].get(ngram, 0))
            if total_count >= min_count:
                filtered_ngrams.add(ngram)
        
        results = {
            'information_gain': [],
            'information_gain_with_length_bonus': [],
            'mutual_information_variable': [],
            'mutual_information_with_length_bonus': []
        }
        
        # Calculate mutual information for variable-length n-grams
        mi_scores = {}
        for ngram in filtered_ngrams:
            count_1_with = sum(1 for seq in self.sequences1 if ngram in seq)
            count_1_without = len(self.sequences1) - count_1_with
            count_2_with = sum(1 for seq in self.sequences2 if ngram in seq)
            count_2_without = len(self.sequences2) - count_2_with
            
            if SCIPY_AVAILABLE:
                contingency = np.array([[count_1_with, count_1_without],
                                      [count_2_with, count_2_without]])
                
                if contingency.sum() > 0:
                    mi = self._mutual_information_from_contingency(contingency)
                    mi_scores[ngram] = mi
                else:
                    mi_scores[ngram] = 0.0
            else:
                # Fallback: use simple frequency-based mutual information
                total_with = count_1_with + count_2_with
                total_without = count_1_without + count_2_without
                total_sequences = len(self.sequences1) + len(self.sequences2)
                
                if total_with > 0 and total_without > 0:
                    p_with = total_with / total_sequences
                    p_without = total_without / total_sequences
                    p1_given_with = count_1_with / total_with
                    p2_given_with = count_2_with / total_with
                    p1_given_without = count_1_without / total_without
                    p2_given_without = count_2_without / total_without
                    
                    mi = 0
                    if p1_given_with > 0:
                        mi += p_with * p1_given_with * math.log2(p1_given_with / (len(self.sequences1) / total_sequences))
                    if p2_given_with > 0:
                        mi += p_with * p2_given_with * math.log2(p2_given_with / (len(self.sequences2) / total_sequences))
                    if p1_given_without > 0:
                        mi += p_without * p1_given_without * math.log2(p1_given_without / (len(self.sequences1) / total_sequences))
                    if p2_given_without > 0:
                        mi += p_without * p2_given_without * math.log2(p2_given_without / (len(self.sequences2) / total_sequences))
                    
                    mi_scores[ngram] = mi
                else:
                    mi_scores[ngram] = 0.0
        
        # Rank by information gain
        ig_sorted = sorted([(ngram, score) for ngram, score in info_gain.items() 
                           if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['information_gain'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
                                      for ngram, score in ig_sorted[:top_k]]
        
        # Rank by information gain with length bonus
        ig_length_sorted = sorted([(ngram, self.calculate_length_penalty_score(ngram, score, length_bonus)) 
                                  for ngram, score in info_gain.items() 
                                  if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['information_gain_with_length_bonus'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
                                                        for ngram, score in ig_length_sorted[:top_k]]
        
        # Rank by mutual information
        mi_sorted = sorted([(ngram, score) for ngram, score in mi_scores.items() 
                           if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['mutual_information_variable'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
                                                 for ngram, score in mi_sorted[:top_k]]
        
        # Rank by mutual information with length bonus
        mi_length_sorted = sorted([(ngram, self.calculate_length_penalty_score(ngram, score, length_bonus)) 
                                  for ngram, score in mi_scores.items() 
                                  if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['mutual_information_with_length_bonus'] = [(ngram, score, self.variable_ngrams['lengths'][ngram]) 
                                                          for ngram, score in mi_length_sorted[:top_k]]
        
        return results
    
    def find_distinctive_patterns(self, n: int, top_k: int = 20, min_count: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Find distinctive n-gram patterns for each dataset."""
        self.count_ngrams(n)
        
        # Calculate various metrics
        freq1, freq2 = self.calculate_frequencies(n)
        mi_scores = self.calculate_mutual_information(n)
        chi2_results = self.calculate_chi_square(n, min_count)
        log_odds = self.calculate_log_odds_ratio(n)
        
        # Filter n-grams by minimum count
        filtered_ngrams = set()
        for ngram in set(self.ngram_counts1[n].keys()) | set(self.ngram_counts2[n].keys()):
            total_count = self.ngram_counts1[n].get(ngram, 0) + self.ngram_counts2[n].get(ngram, 0)
            if total_count >= min_count:
                filtered_ngrams.add(ngram)
        
        # Rank patterns by different metrics
        results = {
            'mutual_information': [],
            'chi_square': [],
            'log_odds_ratio_dataset1': [],
            'log_odds_ratio_dataset2': [],
            'frequency_ratio_dataset1': [],
            'frequency_ratio_dataset2': []
        }
        
        # Mutual Information ranking
        mi_sorted = sorted([(ngram, score) for ngram, score in mi_scores.items() 
                           if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        results['mutual_information'] = mi_sorted[:top_k]
        
        # Chi-square ranking (only if scipy is available)
        if chi2_results:
            chi2_sorted = sorted([(ngram, chi2) for ngram, (chi2, p_val) in chi2_results.items() 
                                 if ngram in filtered_ngrams and p_val < 0.05], 
                                key=lambda x: x[1], reverse=True)
            results['chi_square'] = chi2_sorted[:top_k]
        
        # Log odds ratio ranking (positive for dataset1, negative for dataset2)
        log_odds_sorted = sorted([(ngram, score) for ngram, score in log_odds.items() 
                                 if ngram in filtered_ngrams], key=lambda x: x[1], reverse=True)
        
        # Split by positive/negative log odds
        positive_odds = [(ngram, score) for ngram, score in log_odds_sorted if score > 0]
        negative_odds = [(ngram, abs(score)) for ngram, score in log_odds_sorted if score < 0]
        
        results['log_odds_ratio_dataset1'] = positive_odds[:top_k]
        results['log_odds_ratio_dataset2'] = negative_odds[:top_k]
        
        # Frequency ratio ranking
        freq_ratios_1 = []
        freq_ratios_2 = []
        
        for ngram in filtered_ngrams:
            f1 = freq1.get(ngram, 0)
            f2 = freq2.get(ngram, 0)
            
            if f2 > 0:
                ratio_1 = f1 / f2
                freq_ratios_1.append((ngram, ratio_1))
            
            if f1 > 0:
                ratio_2 = f2 / f1
                freq_ratios_2.append((ngram, ratio_2))
        
        results['frequency_ratio_dataset1'] = sorted(freq_ratios_1, key=lambda x: x[1], reverse=True)[:top_k]
        results['frequency_ratio_dataset2'] = sorted(freq_ratios_2, key=lambda x: x[1], reverse=True)[:top_k]
        
        return results
    
    def print_analysis_results(self, n: int, top_k: int = 20, min_count: int = 3):
        """Print comprehensive analysis results."""
        print(f"\n{'='*60}")
        print(f"N-GRAM ANALYSIS RESULTS (n={n})")
        print(f"{'='*60}")
        print(f"Dataset 1 ({self.label1}): {len(self.sequences1)} sequences")
        print(f"Dataset 2 ({self.label2}): {len(self.sequences2)} sequences")
        print(f"Minimum count threshold: {min_count}")
        
        results = self.find_distinctive_patterns(n, top_k, min_count)
        
        # Print results for each metric
        for metric, patterns in results.items():
            if not patterns:
                continue
                
            print(f"\n{'-'*50}")
            print(f"TOP PATTERNS BY {metric.upper().replace('_', ' ')}")
            print(f"{'-'*50}")
            
            for i, (ngram, score) in enumerate(patterns, 1):
                count1 = self.ngram_counts1[n].get(ngram, 0)
                count2 = self.ngram_counts2[n].get(ngram, 0)
                
                print(f"{i:2d}. '{ngram}' (score: {score:.4f})")
                print(f"    {self.label1}: {count1:4d} | {self.label2}: {count2:4d}")
        
        # Print summary statistics
        print(f"\n{'-'*50}")
        print("SUMMARY STATISTICS")
        print(f"{'-'*50}")
        print(f"Total {n}-grams in {self.label1}: {self.total_ngrams1}")
        print(f"Total {n}-grams in {self.label2}: {self.total_ngrams2}")
        print(f"Unique {n}-grams in {self.label1}: {len(self.ngram_counts1[n])}")
        print(f"Unique {n}-grams in {self.label2}: {len(self.ngram_counts2[n])}")
        
        # Overlap statistics
        common_ngrams = set(self.ngram_counts1[n].keys()) & set(self.ngram_counts2[n].keys())
        unique_to_1 = set(self.ngram_counts1[n].keys()) - set(self.ngram_counts2[n].keys())
        unique_to_2 = set(self.ngram_counts2[n].keys()) - set(self.ngram_counts1[n].keys())
        
        print(f"Common {n}-grams: {len(common_ngrams)}")
        print(f"Unique to {self.label1}: {len(unique_to_1)}")
        print(f"Unique to {self.label2}: {len(unique_to_2)}")
    
    def print_dynamic_analysis_results(self, min_len: int = 1, max_len: int = 10, top_k: int = 20, 
                                     min_count: int = 3, length_bonus: float = 0.1):
        """Print comprehensive dynamic n-gram analysis results."""
        print(f"\n{'='*70}")
        print(f"DYNAMIC N-GRAM ANALYSIS RESULTS (lengths {min_len}-{max_len})")
        print(f"{'='*70}")
        print(f"Dataset 1 ({self.label1}): {len(self.sequences1)} sequences")
        print(f"Dataset 2 ({self.label2}): {len(self.sequences2)} sequences")
        print(f"Length range: {min_len}-{max_len}")
        print(f"Minimum count threshold: {min_count}")
        print(f"Length bonus factor: {length_bonus}")
        
        results = self.find_dynamic_patterns(min_len, max_len, top_k, min_count, length_bonus)
        
        # Print results for each metric
        for metric, patterns in results.items():
            if not patterns:
                continue
                
            print(f"\n{'-'*60}")
            print(f"TOP PATTERNS BY {metric.upper().replace('_', ' ')}")
            print(f"{'-'*60}")
            
            for i, (ngram, score, length) in enumerate(patterns, 1):
                count1 = self.variable_ngrams['dataset1'].get(ngram, 0)
                count2 = self.variable_ngrams['dataset2'].get(ngram, 0)
                
                print(f"{i:2d}. '{ngram}' (length: {length}, score: {score:.4f})")
                print(f"    {self.label1}: {count1:4d} | {self.label2}: {count2:4d}")
        
        # Print summary statistics
        print(f"\n{'-'*60}")
        print("SUMMARY STATISTICS")
        print(f"{'-'*60}")
        
        total_ngrams_1 = sum(self.variable_ngrams['dataset1'].values())
        total_ngrams_2 = sum(self.variable_ngrams['dataset2'].values())
        
        print(f"Total n-grams in {self.label1}: {total_ngrams_1}")
        print(f"Total n-grams in {self.label2}: {total_ngrams_2}")
        print(f"Unique n-grams in {self.label1}: {len(self.variable_ngrams['dataset1'])}")
        print(f"Unique n-grams in {self.label2}: {len(self.variable_ngrams['dataset2'])}")
        
        # Length distribution
        length_dist = {}
        for ngram, length in self.variable_ngrams['lengths'].items():
            length_dist[length] = length_dist.get(length, 0) + 1
        
        print(f"\nLength distribution:")
        for length in sorted(length_dist.keys()):
            print(f"  Length {length}: {length_dist[length]} unique n-grams")
        
        # Overlap statistics
        common_ngrams = set(self.variable_ngrams['dataset1'].keys()) & set(self.variable_ngrams['dataset2'].keys())
        unique_to_1 = set(self.variable_ngrams['dataset1'].keys()) - set(self.variable_ngrams['dataset2'].keys())
        unique_to_2 = set(self.variable_ngrams['dataset2'].keys()) - set(self.variable_ngrams['dataset1'].keys())
        
        print(f"\nOverlap analysis:")
        print(f"Common n-grams: {len(common_ngrams)}")
        print(f"Unique to {self.label1}: {len(unique_to_1)}")
        print(f"Unique to {self.label2}: {len(unique_to_2)}")


def load_sequences_from_json(file_path: str) -> List[str]:
    """Load sequences from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return [str(item) for item in data]
        else:
            raise ValueError(f"Expected a list in JSON file, got {type(data)}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def main():
    """Main function to run the n-gram analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze n-gram patterns between two sets of model responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis.py data/sequence/reasoning.json data/sequence/non_reasoning.json -n 3 -k 15
  python analysis.py file1.json file2.json -n 2 4 -k 20 --min-count 5
  python analysis.py reasoning.json non_reasoning.json -n 1 2 3 4 5 --labels "Reasoning" "Non-Reasoning"
        """
    )
    
    parser.add_argument('file1', help='Path to the first JSON file containing sequences')
    parser.add_argument('file2', help='Path to the second JSON file containing sequences')
    parser.add_argument('-n', '--ngram-sizes', type=int, nargs='+', default=[2, 3, 4],
                       help='N-gram sizes to analyze (default: 2 3 4)')
    parser.add_argument('-k', '--top-k', type=int, default=20,
                       help='Number of top patterns to show for each metric (default: 20)')
    parser.add_argument('--min-count', type=int, default=3,
                       help='Minimum count threshold for n-grams (default: 3)')
    parser.add_argument('--labels', nargs=2, default=['Dataset1', 'Dataset2'],
                       help='Labels for the two datasets (default: Dataset1 Dataset2)')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic n-gram analysis that encourages longer n-grams')
    parser.add_argument('--min-length', type=int, default=1,
                       help='Minimum n-gram length for dynamic analysis (default: 1)')
    parser.add_argument('--max-length', type=int, default=10,
                       help='Maximum n-gram length for dynamic analysis (default: 10)')
    parser.add_argument('--length-bonus', type=float, default=0.1,
                       help='Length bonus factor for encouraging longer n-grams (default: 0.1)')
    
    args = parser.parse_args()
    
    try:
        # Load sequences
        print("Loading sequences...")
        sequences1 = load_sequences_from_json(args.file1)
        sequences2 = load_sequences_from_json(args.file2)
        
        print(f"Loaded {len(sequences1)} sequences from {args.file1}")
        print(f"Loaded {len(sequences2)} sequences from {args.file2}")
        
        # Initialize analyzer
        analyzer = NGramAnalyzer(sequences1, sequences2, args.labels[0], args.labels[1])
        
        # Run analysis
        if args.dynamic:
            # Dynamic n-gram analysis
            analyzer.print_dynamic_analysis_results(
                args.min_length, args.max_length, args.top_k, 
                args.min_count, args.length_bonus
            )
        else:
            # Traditional fixed-length n-gram analysis
            for n in args.ngram_sizes:
                analyzer.print_analysis_results(n, args.top_k, args.min_count)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
