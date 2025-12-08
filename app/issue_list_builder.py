# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
import json

ISSUE_LIST_ROOT = Path(r"C:\my_games\shorts_make\issue_list")


def _create_driver():
    """
    Selenium Chrome 드라이버 생성 (headless).
    - 이 모듈 안에서만 사용.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

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
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup

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

    _prog(f"네이버 기반 원시 토픽 총 {len(topics)}개 수집 완료", stage="done")
    print(f"🧩 네이버 기반 원시 토픽 총 {len(topics)}개 수집 완료")
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
