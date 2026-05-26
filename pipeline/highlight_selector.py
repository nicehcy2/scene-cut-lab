"""
점수 기준으로 장면 선택.
- keep은 모두 포함
- maybe는 final_score >= maybe_min_score 인 경우만 포함
- --top-n은 최대 개수 상한선 역할 (0이면 제한 없음)
"""
from typing import List

import config


def select_top(results: List[dict], top_n: int, keep_only: bool = False) -> List[dict]:
    def _include(r: dict) -> bool:
        decision = r.get("decision", "drop")
        if decision == "keep":
            return True
        if not keep_only and decision == "maybe":
            return r.get("final_score", 0) >= config.MAYBE_MIN_SCORE
        return False

    candidates = [r for r in results if _include(r)]

    # final_score 내림차순 정렬
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # top_n이 0이면 전체 포함, 아니면 상한 적용
    if top_n > 0:
        candidates = candidates[:top_n]

    # 출력 시 시간순 정렬
    candidates.sort(key=lambda x: x["start"])
    return candidates
