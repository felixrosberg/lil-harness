"""Run with: python test_stats.py"""

from stats import mean, median

assert mean([2, 4, 6]) == 4, f"mean([2, 4, 6]) should be 4, got {mean([2, 4, 6])}"
assert median([3, 1, 2]) == 2
assert median([4, 1, 3, 2]) == 2.5
print("all tests passed")
