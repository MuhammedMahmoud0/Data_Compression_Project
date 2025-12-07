import math
from collections import Counter


def calculate_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total_chars
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_avg_length_formula(text, codes_map):
    if not text:
        return 0.0
    counts = Counter(text)
    total_chars = len(text)
    avg_len = 0.0
    for char, count in counts.items():
        p = count / total_chars
        bit_len = codes_map.get(char, 0)
        avg_len += p * bit_len
    return avg_len


def build_frequency(text):
    return Counter(text)
