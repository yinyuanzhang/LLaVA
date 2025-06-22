import collections
import matplotlib.pyplot as plt
import numpy as np # Often useful with matplotlib

class CacheStatisticsCollector:
    """
    A dedicated class to collect, report, and visualize statistics on background and object
    token distribution during cache-enabled inference.
    """
    def __init__(self, total_expected_tokens=576):
        self.total_expected_tokens = total_expected_tokens
        self.all_background_token_counts = []
        self.all_object_token_counts = []
        self.background_token_distribution = collections.defaultdict(int) # Counts frequency of each bg token number
        self.num_processed_samples = 0
        
        # --- New: Cache hit/miss counters ---
        self.cache_hits = 0
        self.cache_misses = 0

    def collect_stats(self, background_valid_count, object_valid_count):
        """
        Collects statistics for a single sample.
        Args:
            background_valid_count (int): Number of valid background tokens for the current sample.
            object_valid_count (int): Number of valid object tokens for the current sample.
        """
        self.all_background_token_counts.append(background_valid_count)
        self.all_object_token_counts.append(object_valid_count)
        self.background_token_distribution[background_valid_count] += 1
        self.num_processed_samples += 1

    # --- New method to update hit/miss ---
    def record_cache_outcome(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def report_stats(self, dataset, plot_filename="./background_token_distribution.png"):
        """
        Calculates and prints the accumulated statistics, and optionally plots the distribution.
        Args:
            plot_filename (str): The filename to save the background token distribution plot.
                                 If None, the plot will not be saved.
        """
        print("\n--- 缓存统计结果 ---")

        if self.num_processed_samples == 0:
            print("没有处理任何图片，无法生成统计数据。")
            return

        plot_filename = f"./{dataset}_background_token_distribution.png"
        # Calculate average token counts and percentages
        average_bg_tokens = sum(self.all_background_token_counts) / self.num_processed_samples
        average_obj_tokens = sum(self.all_object_token_counts) / self.num_processed_samples
        
        average_bg_percentage = (average_bg_tokens / self.total_expected_tokens) * 100
        average_obj_percentage = (average_obj_tokens / self.total_expected_tokens) * 100

        print(f"总处理图片数量: {self.num_processed_samples}")
        print(f"平均每张图片**背景Token数量**: {average_bg_tokens:.2f}")
        print(f"平均每张图片**目标Token数量**: {average_obj_tokens:.2f}")
        print(f"平均**背景Token占比**: {average_bg_percentage:.2f}%")
        print(f"平均**目标Token占比**: {average_obj_percentage:.2f}%")

        # Print distribution of different background token counts
        print("\n**不同背景Token数量的统计分布**:")
        for token_count in sorted(self.background_token_distribution.keys()):
            frequency = self.background_token_distribution[token_count]
            percentage_of_samples = (frequency / self.num_processed_samples) * 100
            print(f"  背景Token数 {token_count}: 出现 {frequency} 次 ({percentage_of_samples:.2f}%)")

        # --- Cache Hit Rate Calculation ---
        total_cache_attempts = self.cache_hits + self.cache_misses
        if total_cache_attempts > 0:
            hit_rate = (self.cache_hits / total_cache_attempts) * 100
            print(f"\n缓存命中统计 (针对背景特征检索):")
            print(f"  总尝试次数: {total_cache_attempts}")
            print(f"  命中次数: {self.cache_hits}")
            print(f"  未命中次数: {self.cache_misses}")
            print(f"  **缓存命中率: {hit_rate:.2f}%** (阈值: {self.model_cache_distance_threshold if hasattr(self, 'model_cache_distance_threshold') else 'N/A'})")
        else:
            print("\n没有进行背景缓存检索，无法计算命中率。")

        # --- Plotting Background Token Distribution ---
        if plot_filename and self.num_processed_samples > 0:
            print(f"\n绘制背景Token分布图并保存到: {plot_filename}")
            counts = sorted(self.background_token_distribution.keys())
            frequencies = [self.background_token_distribution[c] for c in counts]
            
            plt.figure(figsize=(12, 6))
            plt.bar(counts, frequencies, width=1.0, align='center', color='skyblue', edgecolor='black')
            
            plt.xlabel("Background token num")
            plt.ylabel("Sample num")
            plt.title("Background distribution")
            plt.xticks(counts, rotation=90 if len(counts) > 20 else 0) # Rotate x-axis labels if too many
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plt.savefig(plot_filename, dpi=300)
            plt.close() # Close the plot to free memory




# 统计指标：
# 1. 背景数目统计 count
# 2. 缓存命中率            