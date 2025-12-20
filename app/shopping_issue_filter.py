from pathlib import Path

def load_latest_issue_list() -> dict:
    """issue_list/에서 가장 최근 JSON 하나 골라서 로드"""

def pick_shopping_topics_with_ai(raw_topics: list[dict], ai=None) -> list[dict]:
    """
    raw_topics(뉴스/이슈) 중에서
    - 쇼핑/상품으로 이어질 만한 것만 골라서
    - 정규화된 형태의 '쇼핑 키워드' 리스트로 변환
    예: [{"keyword": "임세령 선글라스", "reason": "...", "sources": [...]} ...]
    """

def save_shopping_topics(topics: list[dict]) -> Path:
    """쇼핑 후보 키워드를 shopping_topics/{날짜}/{시분초}.json 으로 저장"""
