import os
from dotenv import load_dotenv
load_dotenv()

wd_id = os.getenv("WP_ID", "none")
wd_pw = os.getenv("WP_PW", "none")
domain_adress = os.getenv("WP_DOMAIN", "none")
api_key = os.getenv("OPENAI_API_KEY", "none")
my_category = os.getenv("BLOG_CATEGORY", "none")
my_category_list = my_category
fallback_thumb_path = os.getenv("FALLBACK_THUMB_PATH", "")
fallback_scene_path = os.getenv("FALLBACK_SCENE_PATH", "")
my_google_custom_api = os.getenv("GOOGLE_CUSTOM_API", "")

dir_path = "C:\\my_games\\shorts_make\\app\\blog_setting"
# 카테고리 #
file_path_category_list = dir_path + "\\category_list.txt"
# 생활정보 #
file_path_topic = dir_path + "\\my_topic.txt"
file_path_topic_system = dir_path + "\\topic_system.txt"
file_path_topic_user = dir_path + "\\topic_user.txt"
# 이슈 #
file_path_issue = dir_path + "\\my_issue.txt"
file_path_issue_system = dir_path + "\\issue_system.txt"
file_path_issue_user = dir_path + "\\issue_user.txt"

#########
# 카테고리 #
#########

for i in range(3):
    if os.path.isfile(file_path_category_list) == True:
        # 파일 읽기
        with open(file_path_category_list, "r", encoding='utf-8-sig') as file:
            category_list = file.read()
            my_category_list = category_list.splitlines()
            break


    else:
        with open(file_path_category_list, "w", encoding='utf-8-sig') as file:
            file.write("생활 팁과 정보 (Blue)\n이슈 (Yellow)")

#########
# 생활팁 #
#########

for i in range(3):
    if os.path.isfile(file_path_topic) == True:
        # 파일 읽기
        with open(file_path_topic, "r", encoding='utf-8-sig') as file:
            thismytopic_one = file.read()
            my_topic = thismytopic_one
            break


    else:
        with open(file_path_topic, "w", encoding='utf-8-sig') as file:
            file.write("- 주제는 실생활에서 활용도 높은 세금, 지원금, 연금, 공과금, 부동산, 법률, 주식 등의 돈과 관련된 항목을 위주로 설정")

for i in range(3):
    if os.path.isfile(file_path_topic_system) == True:
        # 파일 읽기
        with open(file_path_topic_system, "r", encoding='utf-8-sig') as file:
            topic_system_one = file.read()
            my_topic_system = topic_system_one
            break


    else:
        with open(file_path_topic_system, "w", encoding='utf-8-sig') as file:
            file.write(
                f"""
            주요 독자는 정책·생활지원금·신청제도·절약팁 등 실질적 도움이 되는 콘텐츠를 찾는 일반 대중입니다.
            **주의:** 제목에 '2025년 여름철', '이번 달' 등의 반복적 시점 표현은 제외하고, 정보 중심 키워드로 작성하세요."""
            )



for i in range(3):
    if os.path.isfile(file_path_topic_user) == True:
        # 파일 읽기
        with open(file_path_topic_user, "r", encoding='utf-8-sig') as file:
            topic_user_one = file.read()
            my_topic_user = topic_user_one
            break


    else:
        with open(file_path_topic_user, "w", encoding='utf-8-sig') as file:
            file.write(
                f"""
                - 특히 아래 분야 우선 고려:
                - 부동산 정책, 금융 혜택, 세금 감면, 정부 지원금, 생활 신청제도, 에너지 절약, 소비자 혜택
                """
                )

#########
# 이슈 #
#########

for i in range(3):
    if os.path.isfile(file_path_issue) == True:
        # 파일 읽기
        with open(file_path_issue, "r", encoding='utf-8-sig') as file:
            thismyissue = file.read()
            my_issue = thismyissue
            break


    else:
        with open(file_path_issue, "w", encoding='utf-8-sig') as file:
            file.write("- 주제는 실생활에서 영향을 미칠수 있는 것들과 관련된 항목을 위주로 설정")

for i in range(3):
    if os.path.isfile(file_path_issue_system) == True:
        # 파일 읽기
        with open(file_path_issue_system, "r", encoding='utf-8-sig') as file:
            topic_system_one = file.read()
            my_issue_system = topic_system_one
            break


    else:
        with open(file_path_issue_system, "w", encoding='utf-8-sig') as file:
            file.write(
                f"""
            주요 독자는 자극적인 이슈 및 정보를 콘텐츠를 찾는 일반 대중입니다.
            **주의:** 제목에 '2025년 여름철', '이번 달' 등의 반복적 시점 표현은 제외하고, 정보 중심 키워드로 작성하세요."""
            )

for i in range(3):
    if os.path.isfile(file_path_issue_user) == True:
        # 파일 읽기
        with open(file_path_issue_user, "r", encoding='utf-8-sig') as file:
            topic_user_one = file.read()
            my_issue_user = topic_user_one
            break


    else:
        with open(file_path_issue_user, "w", encoding='utf-8-sig') as file:
            file.write(
                f"""
                - 특히 아래 분야 우선 고려:
                - 현재의 이슈, 정보, 뉴스
                """
            )




