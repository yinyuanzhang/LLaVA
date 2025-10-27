# statistics_collector.py
import collections

class CacheStatisticsCollector:
    def __init__(self):
        self.bg_counts = []
        self.fg_counts = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.write_ops = 0

    def record_counts(self, bg_count, fg_count):
        self.bg_counts.append(bg_count)
        self.fg_counts.append(fg_count)

    def record_cache_outcome(self, hit: bool):
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
    def record_write(self):
        self.write_ops += 1

    def report(self):
        print("\n--- Background Caching Statistics ---")
        total_samples = len(self.bg_counts)
        if total_samples == 0:
            print("No samples processed.")
            return

        avg_bg = sum(self.bg_counts) / total_samples
        avg_fg = sum(self.fg_counts) / total_samples
        print(f"Total Samples: {total_samples}")
        print(f"Average Background Logical Tokens: {avg_bg:.2f}")
        print(f"Average Foreground Logical Tokens: {avg_fg:.2f}")

        total_searches = self.cache_hits + self.cache_misses
        if total_searches > 0:
            hit_rate = (self.cache_hits / total_searches) * 100
            print(f"\nCache Performance:")
            print(f"  Total Searches: {total_searches}")
            print(f"  Hits: {self.cache_hits}")
            print(f"  Misses: {self.cache_misses}")
            print(f"  Hit Rate: {hit_rate:.2f}%")
        
        print(f"Total Write Operations: {self.write_ops}")