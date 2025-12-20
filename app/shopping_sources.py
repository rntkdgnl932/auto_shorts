# -*- coding: utf-8 -*-
from __future__ import annotations

"""
쇼핑/트렌드 외부 소스 수집용 헬퍼 모듈.

- 유튜브 쇼핑/리뷰/쇼츠 트렌드
- 해외 쇼핑 트렌드(Reddit)
- 쿠팡 BEST/급상승 (현재는 스텁)

주의:
- 실제 API 키/토큰은 환경변수 또는 settings.py를 통해 주입하는 것을 전제로 한다.
- 키가 없거나 의존 라이브러리가 없으면, 각 함수는 안전하게 빈 리스트([])를 반환한다.
"""

from typing import List, Dict, Any
import os
import logging

import requests

logger = logging.getLogger(__name__)


# ───────────────────────────────────
# 공통 topic 스키마 예시
# {
#   "source": "youtube_shopping" | "reddit_shopping" | "coupang_best",
#   "title": "사람이 읽을 수 있는 제목/키워드",
#   "rank": 1,
#   "url": "https://...",
#   "extra": {...}
# }
# ───────────────────────────────────


# ───────────────────────────────────
# 1) 유튜브 쇼핑/쇼츠 트렌드
# ───────────────────────────────────

def collect_youtube_shopping_trends(max_items: int = 50) -> List[Dict[str, Any]]:
    """
    YouTube Data API v3 기반 쇼핑/리뷰/언박싱/쇼츠 트렌드 수집.

    반환 포맷:
    {
        "source": "youtube_shopping",
        "title": "영상 제목",
        "rank": 1,
        "url": "https://youtu.be/...",
        "extra": {
            "channel": "채널명",
            "views": 12345,
            "published_at": "ISO8601",
        }
    }

    - YOUTUBE_API_KEY 환경변수 또는 settings.YOUTUBE_API_KEY 를 사용한다.
    - 키가 없으면 빈 리스트를 반환한다.
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")


    # 검색 파라미터 설정
    search_url = "https://www.googleapis.com/youtube/v3/search"
    # 쇼핑 관련 한국어 키워드 묶음
    query = "쇼핑 리뷰 언박싱 하울 추천템 인생템"

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_items, 50),
        "regionCode": "KR",
        "relevanceLanguage": "ko",
        "order": "viewCount",
        "key": api_key,
    }

    try:
        resp = requests.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("[YouTube] 검색 API 호출 실패: %s", e)
        return []

    data = resp.json()
    items = data.get("items", [])
    if not items:
        return []

    # 영상 ID 목록 → 통계(조회수) 한 번에 가져오기
    video_ids = [
        it.get("id", {}).get("videoId")
        for it in items
        if it.get("id", {}).get("videoId")
    ]
    stats_map: Dict[str, Dict[str, Any]] = {}

    if video_ids:
        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        params_v = {
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": api_key,
            "maxResults": len(video_ids),
        }
        try:
            resp_v = requests.get(videos_url, params=params_v, timeout=10)
            resp_v.raise_for_status()
            vdata = resp_v.json()
            for it in vdata.get("items", []):
                vid = it.get("id")
                if not vid:
                    continue
                stats_map[vid] = it.get("statistics", {}) or {}
        except Exception as e:
            logger.warning("[YouTube] videos API 호출 실패(통계 없이 진행): %s", e)

    topics: List[Dict[str, Any]] = []
    for rank, it in enumerate(items, start=1):
        vid = it.get("id", {}).get("videoId")
        snippet = it.get("snippet", {}) or {}

        title = (snippet.get("title") or "").strip()
        if not title:
            continue

        channel = (snippet.get("channelTitle") or "").strip()
        published_at = snippet.get("publishedAt")
        stats = stats_map.get(vid, {}) if vid else {}
        views_str = stats.get("viewCount")
        try:
            views = int(views_str) if views_str is not None else None
        except Exception:
            views = None

        url = f"https://www.youtube.com/watch?v={vid}" if vid else None

        topics.append(
            {
                "source": "youtube_shopping",
                "title": title,
                "rank": rank,
                "url": url,
                "extra": {
                    "channel": channel,
                    "views": views,
                    "published_at": published_at,
                },
            }
        )

    logger.info("[YouTube] 쇼핑/리뷰 트렌드 %d개 수집", len(topics))
    return topics


# ───────────────────────────────────
# 2) 해외 쇼핑 트렌드 (Reddit)
# ───────────────────────────────────

def collect_reddit_shopping_trends(max_items: int = 50) -> List[Dict[str, Any]]:
    """
    Reddit의 쇼핑 관련 서브레딧에서 인기 글을 수집.

    - 인증 없이 public JSON endpoint를 사용 (간단 버전).
    - 네트워크 오류/차단 시 조용히 빈 리스트 반환.

    반환 포맷:
    {
        "source": "reddit_shopping",
        "title": "글 제목",
        "rank": 1,
        "url": "https://www.reddit.com/...",
        "extra": {
            "subreddit": "r/BuyItForLife",
            "ups": 5320,
        }
    }
    """
    subreddits = [
        "BuyItForLife",
        "Shopping",
        "FrugalFemaleFashion",
        "frugal",
        "MakeupAddiction",
    ]

    headers = {
        "User-Agent": "shorts-make-shopping/0.1 (by local-script)"
    }

    collected: List[Dict[str, Any]] = []

    for sub in subreddits:
        if len(collected) >= max_items:
            break

        url = f"https://www.reddit.com/r/{sub}/top.json"
        params = {
            "limit": min(50, max_items),
            "t": "week",  # 최근 1주일 top
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
        except Exception as e:
            logger.warning("[Reddit] %s 호출 실패: %s", sub, e)
            continue

        if resp.status_code != 200:
            logger.warning("[Reddit] %s 응답 코드 %s", sub, resp.status_code)
            continue

        try:
            payload = resp.json()
        except Exception as e:
            logger.warning("[Reddit] %s JSON 파싱 실패: %s", sub, e)
            continue

        for child in payload.get("data", {}).get("children", []):
            data = child.get("data", {}) or {}
            title = (data.get("title") or "").strip()
            if not title:
                continue

            permalink = data.get("permalink") or ""
            post_url = f"https://www.reddit.com{permalink}" if permalink else None
            ups = data.get("ups", 0)

            collected.append(
                {
                    "source": "reddit_shopping",
                    "title": title,
                    "rank": 0,  # 나중에 재정렬
                    "url": post_url,
                    "extra": {
                        "subreddit": f"r/{sub}",
                        "ups": ups,
                    },
                }
            )

            if len(collected) >= max_items:
                break

    # ups 기준 내림차순 정렬 + rank 부여
    collected.sort(key=lambda x: x.get("extra", {}).get("ups", 0), reverse=True)
    for i, item in enumerate(collected, start=1):
        item["rank"] = i

    logger.info("[Reddit] 쇼핑 트렌드 %d개 수집", len(collected))
    return collected


# ───────────────────────────────────
# 3) 쿠팡 BEST/급상승 (현재 스텁)
# ───────────────────────────────────

def collect_coupang_best_and_rising(max_items: int = 50) -> List[Dict[str, Any]]:
    """
    쿠팡 BEST/급상승 상품 수집용 스텁.

    - 현재는 사업자 등록 + 파트너스 API 키가 없으므로 실제 데이터를 가져오지 않는다.
    - 나중에 키를 발급받으면 이 함수 내부만 교체하면 된다.
    - 지금은 구조만 유지하기 위해 항상 빈 리스트를 반환한다.
    """
    logger.info("[Coupang] 아직 API 미구현 상태입니다. 빈 리스트를 반환합니다.")
    return []
