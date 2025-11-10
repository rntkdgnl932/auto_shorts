


import variable as v_

def get_zum_ai_issue_trends():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    from bs4 import BeautifulSoup

    keywords = []
    try:
        options = Options()
        options.add_argument('--headless')  # 필요 시 주석 해제
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://zum.com/")

        # 최대 300초 대기
        WebDriverWait(driver, 300).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "span.issue-word-list__keyword"))
        )

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        keyword_spans = soup.select("span.issue-word-list__keyword")
        for span in keyword_spans:
            text = span.text.strip()
            if text:
                keywords.append(text)

        driver.quit()
    except Exception as e:
        print("⚠️ ZUM AI 이슈 트렌드 오류:", e)
    return keywords



    # 답 받기
    # ai_keywords = get_zum_ai_issue_trends()
    # print("▶ ZUM AI 이슈 트렌드:")
    # for i, kw in enumerate(ai_keywords, 1):
    #     print(f"{i}. {kw}")


def get_google_trending_keywords():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager

    keywords = []

    try:
        options = Options()
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('--headless')  # 필요 시 주석 해제

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://trends.google.co.kr/trends/trendingsearches/daily?geo=KR")

        # 최대 300초 대기
        WebDriverWait(driver, 300).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.mZ3RIc'))
        )

        elements = driver.find_elements(By.CSS_SELECTOR, 'div.mZ3RIc')

        print("\n▶ Google 급상승 키워드:")
        for i, el in enumerate(elements[:20], 1):
            text = el.text.strip()
            if text:
                print(f"{i}. {text}")
                keywords.append(text)

        driver.quit()
    except Exception as e:
        print("⚠️ Google Trends 오류:", e)

    return keywords


    # google_keywords = get_google_trending_keywords()
    # print("▶ Google 급상승 키워드:")
    # for i, kw in enumerate(google_keywords, 1):
    #     print(f"{i}. {kw}")


def get_youtube_trending_titles(max_results=20, region_code="KR"):
    from googleapiclient.discovery import build

    # 카테고리 ID 및 설명 정의
    category_map = {
        "22": "People & Blogs",
        "26": "Howto & Style",
        "24": "News & Politics"
    }

    all_titles = []

    try:
        youtube = build("youtube", "v3", developerKey=v_.my_google_custom_api)

        for category_id, category_name in category_map.items():
            print(f"\n📌 [{category_name}] 카테고리 영상 추출 중...")

            response = youtube.videos().list(
                part="snippet",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=max_results,
                videoCategoryId=category_id
            ).execute()

            category_titles = []
            for item in response.get("items", []):
                title = item["snippet"]["title"].strip()
                if title:
                    category_titles.append(title)
                    all_titles.append(title)

            for i, t in enumerate(category_titles, 1):
                print(f"{i}. {t}")

    except Exception as e:
        print("⚠️ YouTube API 카테고리 트렌딩 오류:", e)

    return all_titles





    # yt_titles = get_youtube_trending_titles()
    # print("▶ YouTube 급상승 영상 제목:")
    # for i, title in enumerate(yt_titles, 1):
    #     print(f"{i}. {title}")




def fetch_health_titles(limit=30):
    import feedparser

    RSS_URL = "https://www.yna.co.kr/rss/health.xml"
    print("▶ 연합뉴스 생활·건강 RSS 불러오는 중...")
    try:
        feed = feedparser.parse(RSS_URL)

        if not feed.entries:
            print("❌ RSS 데이터 없음")
            return []

        print(f"✅ 총 {len(feed.entries)}건 중 상위 {limit}개 제목 추출:\n")

        titles = []
        for i, entry in enumerate(feed.entries[:limit], 1):
            try:
                title = entry.title.strip()
                if title:
                    print(f"{i}. {title}")
                    titles.append(title)
                else:
                    print(f"⚠️ {i}번 항목: 제목 비어 있음")
            except Exception as e:
                print(f"❌ {i}번 항목에서 오류 발생: {e}")
        return titles
    except Exception as e:
        print(f"❌ RSS 파싱 실패: {e}")
        return []


    #fetch_health_titles()


def collect_all_topics():
    topic_list = []

    # 1. ZUM AI 이슈 트렌드
    zum_topics = get_zum_ai_issue_trends()
    topic_list.extend(zum_topics)
    print(f"✅ ZUM 키워드 {len(zum_topics)}개 수집 완료")

    # 2. Google 트렌드
    google_topics = get_google_trending_keywords()
    topic_list.extend(google_topics)
    print(f"✅ Google 키워드 {len(google_topics)}개 수집 완료")

    # 3. YouTube 트렌드 영상 제목
    youtube_topics = get_youtube_trending_titles()
    topic_list.extend(youtube_topics)
    print(f"✅ YouTube 제목 {len(youtube_topics)}개 수집 완료")

    # 4. 연합뉴스 건강 기사 제목

    health_topics = fetch_health_titles()
    topic_list.extend(health_topics)
    print(f"✅ 연합뉴스 건강기사 {len(health_topics)}개 수집 완료")

    return topic_list


def filter_topics_by_category(topic_list):
    from openai import OpenAI
    import json
    import re

    def is_korean(text):
        return bool(re.search(r'[가-힣]', text))

    topic_list = [t for t in topic_list if is_korean(t)][:60]

    client = OpenAI(api_key=v_.api_key, timeout=200)

    """
    OpenAI GPT를 활용해 블로그 카테고리에 맞는 키워드 10개 추출
    """

    system_message = (
        "당신은 콘텐츠 전문가이며, 주어진 블로그 카테고리에 가장 적합한 주제 1개를 JSON 배열 형식으로 출력하는 역할을 맡고 있습니다. "
        "출력은 반드시 JSON 배열만 사용하며, 그 외 부가 설명은 포함하지 마세요. "
        "이모지, 특수문자, 따옴표가 많은 문장, 부제 포함 등은 모두 제외하고 순수한 키워드 문장 10개만 반환하세요. "
        "다음은 출력 예시이며, 이와 같은 형식으로 반환해야합니다." 
        "[\"전기요금 절약하는 5가지 방법\"]"
    )

    user_prompt = f"""
    다음은 뉴스, 트렌드, 유튜브 영상에서 수집된 제목 및 키워드 목록입니다:

    {topic_list}

    이 중 다음 블로그 카테고리와 가장 관련성 높은 주제 10개를 JSON 배열로 출력하세요:

    [카테고리 설명]
    '{v_.my_category}', {v_.my_topic}

    ✅ 조건
    - 한글 제목 중심
    - 실용 정보, 생활 꿀팁, 정책, 정부 지원, 절약 노하우 등 실제로 도움이 되는 주제만
    - 연예, 게임, 스포츠, 유머 영상, 단순 일상 Vlog 등은 제외
    - 특수문자와 이모지 ❌
    - 출력은 반드시 JSON 배열 형식으로 10개의 주제만 포함하세요. (예: ["전기요금 절약법", "폭염 대응 팁", ...])
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )

        raw = response.choices[0].message.content.strip()

        # ✅ "json"이나 ```json ``` 제거
        raw = raw.replace("```json", "").replace("```", "").replace("json", "").strip()

        print("🔍 GPT 응답 원문:", raw)

        filtered = json.loads(raw)
        if isinstance(filtered, list) and len(filtered) == 10:
            return filtered
        else:
            print("⚠️ 10개 배열이 아님:", filtered)
            return []

    except Exception as e:
        print("❌ 필터링 실패:", e)
        return []


def search_naver_blog_top_post(keyword):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    # options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        query = keyword.replace(" ", "+")
        url = f"https://search.naver.com/search.naver?where=view&query={query}&sm=tab_opt"
        driver.get(url)

        # 요소가 로드될 때까지 최대 10초 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a.api_txt_lines"))
        )

        elements = driver.find_elements(By.CSS_SELECTOR, "a.api_txt_lines")
        for el in elements:
            href = el.get_attribute("href")
            if "blog.naver.com" in href:
                print("✅ 최상단 블로그 링크:", href)
                return href

        print("❌ 블로그 링크를 찾을 수 없습니다.")
        return None

    except Exception as e:
        print("❌ 검색 중 오류:", str(e))
        print("🧪 현재 URL:", driver.current_url)
        print("📄 현재 페이지 길이:", len(driver.page_source))
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        return None

    finally:
        driver.quit()




# 예시
# search_naver_blog_top_post("전기요금 할인 제도")



#$ 구글 트렌드
import traceback
import requests
from bs4 import BeautifulSoup
from googlesearch import search


def search_web_for_keyword(keyword: str, num_results: int = 3) -> list[dict] | bool:
    """
    주어진 키워드로 구글 웹 검색을 수행하고, 각 URL에 접속하여
    제목(title)과 요약(snippet)을 추출하여 반환합니다.

    Args:
        keyword (str): 검색할 키워드.
        num_results (int): 처리할 검색 결과의 수. 기본값은 3개.

    Returns:
        list[dict] | bool:
            - 성공 시: 각 결과가 title, link, snippet을 포함하는 딕셔너리 리스트.
            - 실패 시: False를 반환.
    """
    if not keyword:
        print("❌ 에러: 검색어(keyword)가 비어있습니다.")
        return False

    print(f"▶ 웹 검색 실행: '{keyword}' (결과 {num_results}개 요청)")

    try:
        # 1. 구글 검색을 통해 URL 리스트 가져오기 (Generator를 list로 변환)
        # lang="ko"로 한국어 검색 결과를 우선적으로 가져옵니다.
        urls = list(search(keyword, num_results=num_results, lang="ko"))

        if not urls:
            print(f"❌ 검색 결과 없음: '{keyword}'에 대한 검색 결과가 없습니다.")
            return False

        print(f"✅ URL 수집 완료: {len(urls)}개. 이제 각 페이지에서 정보를 추출합니다.")

        formatted_results = []
        # 2. 각 URL에 접속하여 제목과 내용(snippet) 추출
        for url in urls:
            try:
                # 웹사이트가 차단하지 않도록 User-Agent를 설정합니다.
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

                soup = BeautifulSoup(response.text, 'html.parser')

                # 제목 추출 (og:title, 없으면 <title> 태그)
                title = soup.find("meta", property="og:title")
                if title:
                    title = title.get("content")
                else:
                    title = soup.title.string if soup.title else "제목 없음"

                # 요약 추출 (og:description, 없으면 meta description)
                snippet = soup.find("meta", property="og:description")
                if snippet:
                    snippet = snippet.get("content")
                else:
                    snippet = soup.find("meta", attrs={"name": "description"})
                    if snippet:
                        snippet = snippet.get("content")
                    else:
                        # 위 태그들이 없을 경우, 첫 번째 <p> 태그의 텍스트를 일부 사용
                        first_p = soup.find('p')
                        snippet = first_p.get_text() if first_p else "내용 요약 없음"

                formatted_results.append({
                    "title": title.strip(),
                    "link": url,
                    "snippet": snippet.strip()
                })
                print(f"  - 성공: {url}")

            except requests.exceptions.RequestException as e:
                # 특정 페이지 접속 실패 시 건너뛰고 다음 URL 처리
                print(f"  - 실패: {url} (이유: {e})")
                continue  # 다음 루프로 넘어감
            except Exception as e:
                print(f"  - 실패: {url} (알 수 없는 에러: {e})")
                continue

        if not formatted_results:
            print(f"❌ 유효한 검색 결과 없음: 수집한 URL에서 정보를 추출하지 못했습니다.")
            return False

        print(f"✅ 정보 추출 완료: 최종적으로 유효한 결과 {len(formatted_results)}개 수집")
        return formatted_results

    except Exception:
        error_details = traceback.format_exc()
        print(f"❌ 치명적 검색 에러 발생: {error_details}")
        return False

#
# if __name__ == '__main__':
#     # 이 파일(trend_search_page.py)을 직접 실행했을 때만 동작하는 테스트 코드
#     print("--- search_web_for_keyword 함수 테스트 ---")
#
#     test_keyword = "파이썬 블로그 자동화"
#     search_results = search_web_for_keyword(test_keyword, num_results=3)
#
#     if search_results:
#         print(f"\n[테스트 결과: '{test_keyword}']")
#         for i, result in enumerate(search_results, 1):
#             print(f"  {i}. 제목: {result['title']}")
#             print(f"     링크: {result['link']}")
#             print(f"     내용: {result['snippet'][:80]}...")
#             print("-" * 20)
#     else:
#         print(f"\n[테스트 실패] '{test_keyword}'에 대한 결과를 가져오지 못했습니다.")








