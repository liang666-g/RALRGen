"""
本模块提供 BLEU 和 smooth-BLEU 的 Python 实现
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """
    从输入文本片段中提取所有达到指定最大阶数的 n-gram
    参数：
        segment：要从中提取 n-gram 的文本片段
        max_order：此方法返回的 n-gram 的最大词元长度
    返回值：
        包含 segment 中所有达到 max_order 的 n-gram 的计数器
        并统计每个 n-gram 出现的次数
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """
    计算 BLEU 分数
    参数：
        reference_corpus：参考文本列表
        translation_corpus：翻译文本列表
        max_order：最大 n-gram 阶数
        smooth：是否使用平滑方法
    返回值：
        BLEU 分数，n-gram 精确度，简洁惩罚，n-gram 精确度的几何平均值以及简洁性惩罚的三元组
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        # Merge n-grams from all references
        merged_ref_ngrams = collections.Counter()
        for reference in references:
            merged_ref_ngrams |= _get_ngrams(reference, max_order)

        # Candidate n-grams
        translation_ngrams = _get_ngrams(translation, max_order)

        # Count overlapping n-grams
        overlap = translation_ngrams & merged_ref_ngrams
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]

        # Count possible n-grams in candidate
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    if reference_length == 0:
        return (0.0, precisions, 0.0, 0.0, translation_length, reference_length)

    ratio = float(translation_length) / reference_length

    if translation_length == 0:
        bp = 0.0
    elif ratio > 1.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)