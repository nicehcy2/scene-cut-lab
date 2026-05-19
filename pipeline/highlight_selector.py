"""
점수 기준으로 상위 N개 장면 선택
"""
from typing import List


def select_top(results: List[dict], top_n: int) -> List[dict]:
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    top = sorted_results[:top_n]

    # 출력 시 시간순으로 정렬
    top.sort(key=lambda x: x["start"])
    return top
