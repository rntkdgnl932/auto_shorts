# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import re

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
# 공용 저장 유틸
# ─────────────────────────────────────

def _save_issue_list_json(
    topics: List[Dict[str, Any]],
    *,
    suffix: str = "",
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """issue_list/YYYYMMDD/HHMMSS{suffix}.json 저장.

    - suffix 예: "_a", "_b" (선행 언더스코어 포함 권장)
    """
    now = datetime.now()
    date_dir = ISSUE_LIST_ROOT / now.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    file_path = date_dir / f"{now.strftime('%H%M%S')}{suffix}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    msg = f"이슈 리스트 저장 완료: {file_path}"
    if callable(on_progress):
        try:
            on_progress({"stage": "save", "msg": msg})
        except Exception:
            pass

    print(f"💾 {msg}")
    return file_path


def _coerce_topic_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    """최소 스키마 정규화.

    최종 저장 포맷(요구사항):
      {
        "source": "...",
        "title": "...",
        "rank": 10,
        "url": "https://..." | None
      }

    - extra 필드는 있어도 무방하므로 유지한다.
    """
    src = (item.get("source") or "").strip() or "unknown"
    title = (item.get("title") or "").strip()
    rank = item.get("rank", 0)
    try:
        rank = int(rank) if rank is not None else 0
    except Exception:
        rank = 0

    url = item.get("url")
    if url is not None:
        url = str(url).strip() or None

    out: Dict[str, Any] = {
        "source": src,
        "title": title,
        "rank": rank,
        "url": url,
    }

    # extra는 유지(있다면)
    if "extra" in item:
        out["extra"] = item.get("extra")

    return out


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

    file_path = date_dir / f"{now.strftime('%H%M%S')}_a.json"


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
    C:\\my_games\\shorts_make\\issue_list\\YYYYMMDD\\HHMMSS_a.json 으로 저장한다.
    """
    topics = collect_raw_topics_for_shopping_all(on_progress=on_progress)

    # 기존(원시 이슈 풀)은 _a 로 저장
    file_path = _save_issue_list_json(topics, suffix="_a", on_progress=on_progress)
    if callable(on_progress):
        try:
            on_progress({"stage": "done", "msg": f"전체 쇼핑 이슈 리스트(_a) 저장 완료: {file_path}"})
        except Exception:
            pass
    return file_path






def _find_latest_issue_list_file(suffix: str) -> Optional[Path]:
    """issue_list/YYYYMMDD 아래에서 가장 최신의 '*{suffix}.json' 파일을 찾는다.
    suffix 예: '_a', '_b'
    """
    try:
        if not ISSUE_LIST_ROOT.exists():
            return None
        date_dirs = [p for p in ISSUE_LIST_ROOT.iterdir() if p.is_dir()]
        date_dirs.sort(key=lambda p: p.name, reverse=True)
        for d in date_dirs:
            cand = sorted(d.glob(f"*{suffix}.json"), key=lambda p: p.name, reverse=True)
            if cand:
                return cand[0]
    except Exception:
        return None
    return None


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """모델 출력에서 JSON 배열만 최대한 복구해서 파싱한다."""
    if not text:
        return []
    s = text.strip()

    # 코드펜스 제거
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```\s*$", "", s).strip()

    # 1) 전체가 JSON이면 바로
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2) 배열 블록만 추출
    l = s.find("[")
    r = s.rfind("]")
    if l != -1 and r != -1 and r > l:
        block = s[l : r + 1]
        try:
            obj = json.loads(block)
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []


def save_issue_list_for_shopping_ai_b_from_a(
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    a_path: Optional[str] = None,
    max_titles: int = 220,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
) -> Path:
    """
    최신(또는 지정한) _a.json에서 title들을 모아 AI에게 전달하고,
    '이슈와 연관해 매출을 기대할 수 있는 상품 후보' 리스트를 _b.json으로 저장한다.

    저장 최소 포맷:
      { "source": "ai", "title": "...", "rank": 1, "url": null }
    """

    def _prog(msg: str, stage: str = "info"):
        if callable(on_progress):
            try:
                on_progress({"stage": stage, "msg": msg})
            except Exception:
                pass

    # a 파일 결정
    a_file = Path(a_path) if a_path else _find_latest_issue_list_file("_a")
    if not a_file or (not a_file.exists()):
        raise FileNotFoundError("(_a) 이슈 리스트 파일을 찾지 못했습니다. 먼저 1단계로 _a.json을 생성하세요.")

    _prog(f"AI 분석 대상(_a) 로드: {a_file}", stage="load")
    raw = json.loads(a_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("(_a) JSON 구조가 리스트가 아닙니다.")

    titles = []
    for it in raw:
        if isinstance(it, dict):
            t = (it.get("title") or "").strip()
            if t:
                titles.append(t)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    titles = uniq[:max_titles]

    if not titles:
        raise ValueError("(_a)에서 title을 추출하지 못했습니다.")

    _prog(f"AI에 전달할 title 개수: {len(titles)}", stage="prep")

    # Gemini 호출부는 프로젝트에 이미 있는 AI 래퍼(lyrics_gen.py)를 그대로 참조
    try:
        from app.utils import AI, AIConfig
    except Exception as e:
        raise RuntimeError(f"Gemini 요청(AI) 코드(app.lyrics_gen)를 찾지 못했습니다: {e}")

    cfg = AIConfig(provider=provider, gemini_model=model)
    ai = AI(cfg=cfg)

    system = (
        "너는 15년차 쇼핑쇼츠 전문가다. "
        "사람들의 관심을 끌고, 쇼츠로 상품을 소개했을 때 실제 매출로 이어질 가능성이 있는 상품을 찾아내는 능력이 탁월하다."
    )

    bullet = "\n".join([f"- {t}" for t in titles])
    prompt = (
        "아래는 최근 이슈의 제목 목록이다.\n\n"
        f"{bullet}\n\n"
        "요청: 이 제목들과 연관하여 쇼핑쇼츠로 상품을 올렸을 때 매출을 기대할 수 있는 '상품/아이템/키워드' 후보만 골라라.\n"
        "불가능하거나 애매한 것은 과감히 제외해라.\n"
        "출력 형식: JSON 배열만 출력해라. 각 원소는 다음 최소 필드를 포함한다:\n"
        "- source: 반드시 'ai'\n"
        "- title: 상품/아이템/키워드 후보(짧고 명확)\n"
        "- rank: 1부터 증가\n"
        "- url: null\n"
        "가능하면 extra에 products(리스트), reason(짧은 근거), confidence('상/중/하')를 넣어라.\n"
        "JSON 외 다른 텍스트는 절대 출력하지 마라."
    )

    _prog("AI 분석 요청 시작...", stage="ai")
    out_text = ai.ask_smart(system, prompt, prefer="gemini", allow_fallback=False, trace=False)


    _prog("AI 응답 수신 완료. JSON 파싱 중...", stage="ai")
    items = _extract_json_array(out_text)
    if not items:
        _prog("AI 응답에서 JSON 배열을 파싱하지 못했습니다. 빈 리스트로 저장합니다.", stage="warn")
        items = []

    # 스키마 정리 + rank
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = (it.get("title") or "").strip()
        if not title:
            continue

        row = {"source": "ai", "title": title, "rank": 0, "url": None}
        extra = it.get("extra")
        if isinstance(extra, dict) and extra:
            row["extra"] = extra
        cleaned.append(row)

    # 중복 제거 + rank
    seen2 = set()
    final = []
    for it in cleaned:
        k = it["title"]
        if k in seen2:
            continue
        seen2.add(k)
        final.append(it)

    for i, it in enumerate(final, start=1):
        it["rank"] = i

    # 저장(_b)
    now = datetime.now()
    date_dir = ISSUE_LIST_ROOT / now.strftime("%Y%m%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    b_path = date_dir / f"{now.strftime('%H%M%S')}_b.json"
    b_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    _prog(f"AI 상품 후보 리스트(_b) 저장 완료: {b_path}", stage="save")
    return b_path
