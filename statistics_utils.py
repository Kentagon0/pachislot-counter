"""
統計計算ユーティリティモジュール
パチスロ設定判別のための統計計算を提供
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict


def binomial_p_value(n: int, k: int, p: float) -> float:
    """
    二項検定のp値を計算（両側検定）
    
    Args:
        n: 試行回数（総回転数）
        k: 成功回数（小役出現回数）
        p: 理論上の成功確率
    
    Returns:
        p値（両側検定）
    """
    if n <= 0 or p <= 0 or p >= 1:
        return 1.0
    
    # 二項検定（両側）
    result = stats.binomtest(k, n, p, alternative='two-sided')
    return result.pvalue


def calculate_likelihood(n: int, k: int, p: float) -> float:
    """
    尤度を計算
    
    Args:
        n: 試行回数
        k: 成功回数
        p: 確率
    
    Returns:
        尤度（対数尤度のexp）
    """
    if n <= 0 or p <= 0 or p >= 1:
        return 0.0
    
    # 二項分布の確率質量関数
    likelihood = stats.binom.pmf(k, n, p)
    return likelihood


def calculate_relative_likelihood(n: int, k: int, probabilities: List[float]) -> List[float]:
    """
    各設定の相対尤度を計算（正規化）
    
    Args:
        n: 試行回数
        k: 成功回数
        probabilities: 各設定の確率リスト
    
    Returns:
        正規化された尤度リスト（合計1.0）
    """
    likelihoods = [calculate_likelihood(n, k, p) for p in probabilities]
    total = sum(likelihoods)
    
    if total == 0:
        return [1.0 / len(probabilities)] * len(probabilities)
    
    return [l / total for l in likelihoods]


def get_confidence_interval(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    二項分布の信頼区間を計算
    
    Args:
        p: 確率
        n: 試行回数
        confidence: 信頼水準（デフォルト95%）
    
    Returns:
        (下限, 上限) のタプル
    """
    if n <= 0:
        return (0.0, 1.0)
    
    # 正規近似による信頼区間
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / n)
    
    lower = max(0, p - z * se)
    upper = min(1, p + z * se)
    
    return (lower, upper)


def get_expected_count_range(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    期待されるカウント範囲を計算
    
    Args:
        p: 確率
        n: 試行回数
        confidence: 信頼水準
    
    Returns:
        (下限カウント, 上限カウント) のタプル
    """
    lower_p, upper_p = get_confidence_interval(p, n, confidence)
    return (lower_p * n, upper_p * n)


def generate_probability_curve_data(
    probabilities: Dict[str, float], 
    max_n: int,
    step: int = 10
) -> Dict[str, Dict[str, List[float]]]:
    """
    各設定の期待値曲線データを生成
    
    Args:
        probabilities: {設定名: 確率} の辞書
        max_n: 最大試行回数
        step: ステップ幅
    
    Returns:
        グラフ描画用データ
    """
    n_values = list(range(step, max_n + 1, step))
    if max_n not in n_values:
        n_values.append(max_n)
    
    result = {
        "n_values": n_values,
        "curves": {}
    }
    
    for setting_name, p in probabilities.items():
        expected = [p * n for n in n_values]
        lower_bounds = []
        upper_bounds = []
        
        for n in n_values:
            lower, upper = get_expected_count_range(p, n, 0.95)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        result["curves"][setting_name] = {
            "expected": expected,
            "lower": lower_bounds,
            "upper": upper_bounds
        }
    
    return result


def evaluate_setting(p_value: float, significance: float = 0.05) -> Tuple[str, str]:
    """
    p値から設定の評価を判定
    
    Args:
        p_value: p値
        significance: 有意水準
    
    Returns:
        (評価記号, 評価テキスト)
    """
    if p_value < significance / 2:
        return ("✗", "否定的")
    elif p_value < significance:
        return ("△", "やや否定的")
    elif p_value < 0.25:
        return ("○", "可能性あり")
    else:
        return ("◎", "高い可能性")


def parse_probability_input(input_str: str) -> float:
    """
    確率入力をパース（分数または小数対応）
    
    Args:
        input_str: "1/6.5" または "0.1538" または "15.38%"
    
    Returns:
        確率（0-1の範囲）
    """
    input_str = input_str.strip()
    
    # パーセント表記
    if input_str.endswith('%'):
        return float(input_str[:-1]) / 100
    
    # 分数表記 "1/X"
    if '/' in input_str:
        parts = input_str.split('/')
        if len(parts) == 2:
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator != 0:
                return numerator / denominator
    
    # 小数表記
    value = float(input_str)
    # 1より大きい場合は分母として扱う
    if value > 1:
        return 1 / value
    return value
