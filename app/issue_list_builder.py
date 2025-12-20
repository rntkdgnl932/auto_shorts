# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
ISSUE_LIST_ROOT = Path(r"C:\my_games\shorts_make\issue_list")

# ─────────────────────────────────────
# 외부 소스 모듈 임포트
# ─────────────────────────────────────

# 유튜브 / Reddit / 쿠팡 스텁
try:
    from app.shopping_sources import (
        collect_youtube_shopping_trends,
        collect_reddit_shopping_trends,
        collect_coupang_best_and_rising,
    )
except Exception:
    # 문제가 나더라도 전체가 죽지 않도록 안전한 더미 함수 제공
    def collect_youtube_shopping_trends(max_items: int = 50) -> List[Dict[str, Any]]:
        return []

    def collect_reddit_shopping_trends(max_items: int = 50) -> List[Dict[str, Any]]:
        return []

    def collect_coupang_best_and_rising(max_items: int = 50) -> List[Dict[str, Any]]:
        return []

# 네이버 추가 카테고리 + 연합뉴스 트렌드
try:
    from app.blog_trend_search_page import (
        get_naver_news_multi_category_topics,
        get_yonhap_news_trend_topics,
    )
except Exception:
    # blog_trend_search_page 를 못 불러오더라도 전체가 죽지 않게 더미 함수
    def get_naver_news_multi_category_topics(max_items_per_category: int = 30) -> List[Dict[str, Any]]:
        return []

    def get_yonhap_news_trend_topics(max_items_per_category: int = 30) -> List[Dict[str, Any]]:
        return []


# ─────────────────────────────────────
# Selenium (네이버 뉴스 기본 랭킹/경제 용)
# ─────────────────────────────────────

def _create_driver():
    """
    Selenium Chrome 드라이버 생성 (headless).
    - 이 모듈 안에서만 사용.
    """


    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


def _scrape_naver_news_titles(driver, url: str, limit: int = 50) -> List[str]:
    """
    네이버 뉴스 랭킹/경제 섹션 등에서 기사 제목 추출.
    - 여러 CSS 후보를 돌면서 최대한 텍스트를 뽑는다.
    """


    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "body"))
        )
    except Exception:
        pass

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    titles_raw: List[str] = []

    # 1차: 일반 기사 링크 패턴
    for a in soup.select("a[href*='/read?']"):
        text = (a.get_text() or "").strip()
        if not text:
            continue
        if len(text) < 8 or len(text) > 80:
            continue
        titles_raw.append(text)

    # 2차: 랭킹/리스트 전용 클래스 후보들 (네이버 구조 변경 대비)
    css_candidates = [
        "a.rankingnews_link",                # 랭킹뉴스 제목
        "ol.ranking_list a",                 # 랭킹 리스트
        "div.list_body a",                   # 섹션 리스트
        "a[class*='cluster_text_headline']", # 클러스터 헤드라인
        "a[class*='list_title']",            # 리스트 제목
        "a[class*='sa_text_title']",         # 검색/섹션 제목
    ]
    for css in css_candidates:
        for a in soup.select(css):
            text = (a.get_text() or "").strip()
            if not text:
                continue
            if len(text) < 8 or len(text) > 80:
                continue
            titles_raw.append(text)

    # 중복 제거 + limit
    seen = set()
    titles: List[str] = []
    for t in titles_raw:
        if t in seen:
            continue
        seen.add(t)
        titles.append(t)
        if len(titles) >= limit:
            break

    print(f"✅ [NAVER NEWS] {url} 제목 {len(titles)}개 추출")
    return titles


# ─────────────────────────────────────
# 1) 네이버 전용 (랭킹 + 경제)
# ─────────────────────────────────────

def collect_raw_topics_for_shopping_naver(
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    [1단계 전용]
    - 네이버 뉴스 랭킹
    - 네이버 경제 뉴스 리스트

    만 모아서 "쇼핑/이슈용 원시 토픽 풀"을 만든다.
    (AI 판단은 여기서 하지 않음)
    """

    def _prog(msg: str, stage: str = "info"):
        if callable(on_progress):
            try:
                on_progress({"stage": stage, "msg": msg})
            except Exception:
                pass

    topics: List[Dict[str, Any]] = []
    driver = None

    rank_titles: List[str] = []
    econ_titles: List[str] = []

    try:
        _prog("네이버 크롬 드라이버 초기화 중...", stage="init")
        driver = _create_driver()

        # 1) 네이버 뉴스 랭킹
        _prog("네이버 뉴스 랭킹 제목 수집 시작", stage="news_rank")
        rank_url = "https://news.naver.com/main/ranking/popularDay.naver?mid=etc&sid1=111"
        rank_titles = _scrape_naver_news_titles(driver, rank_url, limit=50)
        for idx, title in enumerate(rank_titles, start=1):
            topics.append(
                {
                    "source": "naver_news_rank",
                    "title": title,
                    "rank": idx,
                    "url": None,
                }
            )

        # 2) 네이버 경제 뉴스 리스트
        _prog("네이버 경제 뉴스 제목 수집 시작", stage="news_econ")
        econ_url = "https://news.naver.com/main/list.naver?mode=LSD&mid=sec&sid1=101"
        econ_titles = _scrape_naver_news_titles(driver, econ_url, limit=50)
        for idx, title in enumerate(econ_titles, start=1):
            topics.append(
                {
                    "source": "naver_news_economy",
                    "title": title,
                    "rank": idx,
                    "url": None,
                }
            )

    except Exception as e:
        _prog(f"네이버 이슈 수집 중 오류: {e}", stage="error")
        print(f"⚠ 네이버 이슈 수집 오류: {e}")

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

    total = len(topics)
    msg_detail = (
        f"네이버 뉴스 랭킹 {len(rank_titles)}개, "
        f"경제 뉴스 {len(econ_titles)}개, 총 {total}개 수집 완료"
    )
    _prog(msg_detail, stage="done")
    print(f"🧩 {msg_detail}")
    return topics


def save_issue_list_for_shopping_naver(
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """
    네이버 기반 쇼핑/이슈 토픽들을 모아서
    C:\\my_games\\shorts_make\\issue_list\\YYYYMMDD\\HHMMSS.json 으로 저장한다.
    """
    topics = collect_raw_topics_for_shopping_naver(on_progress=on_progress)

    now = datetime.now()
    date_dir = ISSUE_LIST_ROOT / now.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    file_path = date_dir / f"{now.strftime('%H%M%S')}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    msg = f"네이버 이슈 리스트 저장 완료: {file_path}"
    if callable(on_progress):
        try:
            on_progress({"stage": "save", "msg": msg})
        except Exception:
            pass

    print(f"💾 {msg}")
    return file_path


# ─────────────────────────────────────
# 2) 전체 쇼핑 이슈 (네이버 + 네이버추가 + 쿠팡스텁 + 유튜브 + Reddit + 연합뉴스)
# ─────────────────────────────────────

def collect_raw_topics_for_shopping_all(
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    [1단계 확장 버전]

    - 네이버 이슈(뉴스 랭킹 + 경제)
    - 네이버 추가 카테고리(사회/생활/IT)  ← blog_trend_search_page.get_naver_news_multi_category_topics
    - 쿠팡 BEST/급상승 (현재는 스텁: 항상 빈 리스트)
    - 유튜브 쇼핑/리뷰/쇼츠 트렌드
    - 해외 쇼핑 트렌드(Reddit)
    - 연합뉴스 트렌드/핫/생활/경제/IT

    를 전부 합쳐서 "쇼핑/이슈용 원시 토픽 풀"을 만든다.
    (AI 판단/필터링은 여기서 하지 않음)
    """

    def _prog(msg: str, stage: str = "info") -> None:
        if callable(on_progress):
            try:
                on_progress({"stage": stage, "msg": msg})
            except Exception:
                pass

    topics: List[Dict[str, Any]] = []

    naver_topics: List[Dict[str, Any]] = []
    naver_extra_topics: List[Dict[str, Any]] = []
    coupang_topics: List[Dict[str, Any]] = []
    yt_topics: List[Dict[str, Any]] = []
    rd_topics: List[Dict[str, Any]] = []
    yonhap_topics: List[Dict[str, Any]] = []

    # 1) 네이버 기본(랭킹/경제)
    try:
        _prog("네이버 이슈(뉴스 랭킹/경제) 수집 시작", stage="naver")
        naver_topics = collect_raw_topics_for_shopping_naver(on_progress=on_progress)
        topics.extend(naver_topics)
    except Exception as e:
        _prog(f"네이버 이슈 수집 중 오류: {e}", stage="error")
        print(f"⚠ 네이버 이슈 수집 오류: {e}")

    # 1-추가) 네이버 추가 카테고리(사회/생활/IT) – blog_trend_search_page 사용
    try:
        _prog("네이버 추가 카테고리(사회/생활/IT) 수집 시작", stage="naver_extra")
        naver_extra_topics = get_naver_news_multi_category_topics(max_items_per_category=30)
        topics.extend(naver_extra_topics)
        _prog(f"네이버 추가 카테고리 토픽 {len(naver_extra_topics)}개 수집 완료", stage="naver_extra")
    except Exception as e:
        _prog(f"네이버 추가 카테고리 수집 중 오류: {e}", stage="error")
        print(f"⚠ 네이버 추가 카테고리 수집 오류: {e}")

    # 2) 쿠팡 BEST/급상승 (현재는 스텁: 항상 []).
    try:
        _prog("쿠팡 BEST/급상승(스텁) 수집 시작", stage="coupang")
        coupang_topics = collect_coupang_best_and_rising(max_items=50)
        topics.extend(coupang_topics)
        _prog(f"쿠팡 트렌드 토픽 {len(coupang_topics)}개 수집 (현재 스텁)", stage="coupang")
    except Exception as e:
        _prog(f"쿠팡 트렌드 수집 중 오류: {e}", stage="error")
        print(f"⚠ 쿠팡 트렌드 수집 오류: {e}")

    # 3) 유튜브 쇼핑/쇼츠 트렌드
    try:
        _prog("유튜브 쇼핑/쇼츠 트렌드 수집 시작", stage="youtube")
        yt_topics = collect_youtube_shopping_trends(max_items=50)
        topics.extend(yt_topics)
        _prog(f"유튜브 쇼핑/쇼츠 토픽 {len(yt_topics)}개 수집 완료", stage="youtube")
    except Exception as e:
        _prog(f"유튜브 쇼핑 트렌드 수집 중 오류: {e}", stage="error")
        print(f"⚠ 유튜브 쇼핑 트렌드 수집 오류: {e}")

    # 4) 해외 쇼핑 트렌드(Reddit)
    try:
        _prog("해외 쇼핑 트렌드(Reddit) 수집 시작", stage="reddit")
        rd_topics = collect_reddit_shopping_trends(max_items=50)
        topics.extend(rd_topics)
        _prog(f"Reddit 쇼핑 토픽 {len(rd_topics)}개 수집 완료", stage="reddit")
    except Exception as e:
        _prog(f"Reddit 쇼핑 트렌드 수집 중 오류: {e}", stage="error")
        print(f"⚠ Reddit 쇼핑 트렌드 수집 오류: {e}")

    # 5) 연합뉴스 트렌드(인기/핫/생활/경제/IT)
    try:
        _prog("연합뉴스 트렌드 수집 시작", stage="yonhap")
        yonhap_topics = get_yonhap_news_trend_topics(max_items_per_category=30)
        topics.extend(yonhap_topics)
        _prog(f"연합뉴스 트렌드 토픽 {len(yonhap_topics)}개 수집 완료", stage="yonhap")
    except Exception as e:
        _prog(f"연합뉴스 트렌드 수집 중 오류: {e}", stage="error")
        print(f"⚠ 연합뉴스 트렌드 수집 오류: {e}")

    total = len(topics)
    summary = (
        f"네이버(기본) {len(naver_topics)}개, "
        f"네이버(추가) {len(naver_extra_topics)}개, "
        f"쿠팡 {len(coupang_topics)}개, "
        f"유튜브 {len(yt_topics)}개, "
        f"Reddit {len(rd_topics)}개, "
        f"연합뉴스 {len(yonhap_topics)}개 → 총 {total}개 수집 완료"
    )
    _prog(summary, stage="done")
    print(f"🧩 {summary}")
    return topics


def save_issue_list_for_shopping_all(
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """
    네이버 + 네이버추가 + 유튜브 + Reddit(+쿠팡 스텁) + 연합뉴스 기반
    쇼핑/이슈 토픽들을 모아서
    C:\\my_games\\shorts_make\\issue_list\\YYYYMMDD\\HHMMSS.json 으로 저장한다.
    """
    topics = collect_raw_topics_for_shopping_all(on_progress=on_progress)

    now = datetime.now()
    date_dir = ISSUE_LIST_ROOT / now.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    file_path = date_dir / f"{now.strftime('%H%M%S')}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    msg = f"전체 쇼핑 이슈 리스트 저장 완료: {file_path}"
    if callable(on_progress):
        try:
            on_progress({"stage": "save", "msg": msg})
        except Exception:
            pass

    print(f"💾 {msg}")
    return file_path


