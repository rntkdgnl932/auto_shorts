# youtube_ui.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta, timezone
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QDesktopServices
from PyQt5.QtCore import QUrl, QThread, pyqtSignal

# API 연동 라이브러리 확인
try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

# settings 로드
try:
    from app import settings
except ImportError:
    settings = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. 백그라운드 워커 (수정됨: 여행/교육만 예외처리)
# ─────────────────────────────────────────────────────────────────────────────
class YoutubeWorker(QThread):
    finished = pyqtSignal(list, str)  # 결과 리스트, 메시지

    def __init__(self, api_key, keyword, max_subscribers, date_filter, max_results, duration_filter, category_id):
        super().__init__()
        self.api_key = api_key
        self.keyword = keyword
        self.max_subscribers = max_subscribers
        self.date_filter = date_filter
        self.max_results = max_results
        self.duration_filter = duration_filter
        self.category_id = category_id

    def run(self):
        if not build:
            self.finished.emit([], "google-api-python-client 라이브러리가 설치되지 않았습니다.")
            return

        if not self.api_key:
            self.finished.emit([], "설정(.env)에 YOUTUBE_API_KEY가 없습니다.")
            return

        try:
            youtube = build('youtube', 'v3', developerKey=self.api_key)

            # 1. 날짜 필터 계산 (UTC 표준)
            published_after = None
            now_utc = datetime.now(timezone.utc)
            if self.date_filter == 'day':
                published_after = (now_utc - timedelta(days=1)).isoformat().replace("+00:00", "Z")
            elif self.date_filter == 'week':
                published_after = (now_utc - timedelta(weeks=1)).isoformat().replace("+00:00", "Z")
            elif self.date_filter == 'month':
                published_after = (now_utc - timedelta(days=30)).isoformat().replace("+00:00", "Z")

            video_items = []
            block_keywords = ["Live", "라이브", "스트리밍", "실시간", "M/V", "Music", "Official", "Game", "게임", "리그", "News",
                              "뉴스"]

            # 2. 로직 분기 (사용자 요청 반영: 여행/교육만 Search API로 우회)
            # - 검색어가 있으면: 무조건 Search API
            # - 검색어가 없으면:
            #    - 여행(19), 교육(27) -> MostPopular 미지원(404)이므로 Search API 사용
            #    - 그 외(노하우, 동물 등) -> Videos API (MostPopular) 사용 (기존 유지)

            force_search_categories = ['19', '27']  # 여행/이벤트, 교육

            use_search_api = False
            if self.keyword.strip():
                use_search_api = True
            elif self.category_id in force_search_categories:
                use_search_api = True  # 404 방지용 강제 전환

            if use_search_api:
                # [A] 검색 API 사용 (비용 100)
                # 파라미터 충돌 방지를 위해 kwargs 사용하지 않고 명시적 호출

                req = youtube.search().list(
                    part='snippet',
                    q=self.keyword.strip(),  # 비어있으면 전체 검색 효과
                    type='video',
                    order='viewCount',
                    maxResults=self.max_results,
                    regionCode='KR',
                    videoCategoryId=self.category_id if self.category_id else None,
                    publishedAfter=published_after,
                    videoDuration=self.duration_filter if self.duration_filter != 'any' else None
                )
                res = req.execute()
                items = res.get('items', [])
            else:
                # [B] 인급동(Hot) 모드 (비용 1) - 노하우/스타일 등 정상 카테고리는 여기로 옴
                # videos().list 호출
                req = youtube.videos().list(
                    part='snippet,statistics',
                    chart='mostPopular',
                    regionCode='KR',
                    maxResults=self.max_results,
                    videoCategoryId=self.category_id if self.category_id else ''  # None이면 에러날 수 있어 빈문자열 처리
                )
                res = req.execute()
                items = res.get('items', [])

            # 통계
            cnt_fetched = len(items)
            cnt_filtered_noise = 0
            cnt_filtered_topic = 0

            # 3. 데이터 가공 및 1차 필터링
            for item in items:
                # ID 추출
                if isinstance(item['id'], dict):
                    vid_id = item['id'].get('videoId')
                else:
                    vid_id = item['id']

                if not vid_id: continue

                snippet = item['snippet']
                title = snippet['title']
                ch_title = snippet['channelTitle']

                # [필터] Topic / VEVO
                if ' - Topic' in ch_title or 'VEVO' in ch_title:
                    cnt_filtered_topic += 1
                    continue

                # [필터] 노이즈 키워드
                if any(bk.lower() in title.lower() for bk in block_keywords):
                    cnt_filtered_noise += 1
                    continue

                v_obj = {
                    'video_id': vid_id,
                    'title': title,
                    'channel_id': snippet['channelId'],
                    'channel_title': ch_title,
                    'publish_date': snippet['publishedAt'],
                    'thumb': snippet['thumbnails']['medium']['url'],
                }

                if 'statistics' in item:
                    v_obj['view_count'] = int(item['statistics'].get('viewCount', 0))

                video_items.append(v_obj)

            if not video_items:
                reason = f"API로 {cnt_fetched}개를 가져왔으나 필터링됨.\n"
                reason += f"(음악/Topic: {cnt_filtered_topic}, 게임/노이즈: {cnt_filtered_noise})\n"
                reason += "팁: 검색어나 기간 설정을 변경해보세요."
                self.finished.emit([], reason)
                return

            # 4. 조회수 채우기 (Search 모드였을 때만)
            vid_ids = [v['video_id'] for v in video_items if 'view_count' not in v]
            if vid_ids:
                v_res = youtube.videos().list(
                    part='statistics',
                    id=','.join(vid_ids)
                ).execute()
                stats_map = {item['id']: int(item['statistics'].get('viewCount', 0)) for item in v_res['items']}
                for v in video_items:
                    if v['video_id'] in stats_map:
                        v['view_count'] = stats_map[v['video_id']]

            # 5. 채널 구독자 가져오기
            chan_ids = list({v['channel_id'] for v in video_items})
            chan_stats_map = {}
            for i in range(0, len(chan_ids), 50):
                chunk = chan_ids[i:i + 50]
                c_res = youtube.channels().list(
                    part='statistics',
                    id=','.join(chunk)
                ).execute()
                for item in c_res['items']:
                    sub = item['statistics'].get('subscriberCount')
                    chan_stats_map[item['id']] = int(sub) if sub else 0

            # 6. 성과율 계산 및 구독자 필터
            final_results = []
            cnt_sub_filtered = 0

            for v in video_items:
                sub_count = chan_stats_map.get(v['channel_id'], 0)
                view_count = v.get('view_count', 0)

                if self.max_subscribers > 0 and sub_count > self.max_subscribers:
                    cnt_sub_filtered += 1
                    continue

                ratio = 0.0
                if sub_count > 0:
                    ratio = (view_count / sub_count) * 100

                v['sub_count'] = sub_count
                v['ratio'] = ratio
                final_results.append(v)

            final_results.sort(key=lambda x: x['ratio'], reverse=True)

            if not final_results:
                reason = f"구독자 수 제한({self.max_subscribers}명)으로 {cnt_sub_filtered}개가 제외되었습니다."
                self.finished.emit([], reason)
            else:
                self.finished.emit(final_results, "")

        except Exception as e:
            err_str = str(e)
            if "404" in err_str:
                self.finished.emit([],
                                   "⚠️ 404 오류: 이 카테고리는 인기차트를 지원하지 않습니다.\n개발자 수정사항: 해당 ID를 force_search_categories에 추가해야 합니다.")
            else:
                self.finished.emit([], f"API 오류:\n{err_str}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. 이미지 로더
# ─────────────────────────────────────────────────────────────────────────────
class ImageLoader(QThread):
    loaded = pyqtSignal(int, QPixmap)

    def __init__(self, row, url):
        super().__init__()
        self.row = row
        self.url = url

    def run(self):
        try:
            import requests
            data = requests.get(self.url, timeout=5).content
            pix = QPixmap()
            pix.loadFromData(data)
            self.loaded.emit(self.row, pix)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. 리스트 아이템 위젯
# ─────────────────────────────────────────────────────────────────────────────
class VideoItemWidget(QtWidgets.QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.video_url = f"https://www.youtube.com/watch?v={data['video_id']}"

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)

        # 썸네일
        self.lbl_thumb = QtWidgets.QLabel()
        self.lbl_thumb.setFixedSize(160, 90)
        self.lbl_thumb.setStyleSheet("background-color: #ddd; border: 1px solid #999;")
        self.lbl_thumb.setScaledContents(True)
        lay.addWidget(self.lbl_thumb)

        # 정보
        info_lay = QtWidgets.QVBoxLayout()

        self.btn_title = QtWidgets.QPushButton(data['title'])
        self.btn_title.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_title.setStyleSheet("text-align: left; font-weight: bold; font-size: 14px; border: none; color: #000;")
        self.btn_title.clicked.connect(self.open_link)
        info_lay.addWidget(self.btn_title)

        sub_k = f"{data['sub_count'] / 10000:.1f}만" if data['sub_count'] >= 10000 else f"{data['sub_count']:,}"
        view_k = f"{data['view_count'] / 10000:.1f}만" if data['view_count'] >= 10000 else f"{data['view_count']:,}"
        pub_date = data['publish_date'][:10]

        lbl_meta = QtWidgets.QLabel(f"📺 {data['channel_title']}  |  📅 {pub_date}  |  구독: {sub_k}  |  조회: {view_k}")
        lbl_meta.setStyleSheet("color: #555; font-size: 12px;")
        info_lay.addWidget(lbl_meta)

        lay.addLayout(info_lay)

        # 성과율
        ratio = data['ratio']
        color = "#d32f2f" if ratio >= 300 else ("#1976d2" if ratio >= 100 else "#666")
        lbl_ratio = QtWidgets.QLabel(f"성과율\n{ratio:,.0f}%")
        lbl_ratio.setAlignment(QtCore.Qt.AlignCenter)
        lbl_ratio.setFixedWidth(80)
        lbl_ratio.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color}; border: 1px solid {color}; border-radius: 5px; padding: 5px;")
        lay.addWidget(lbl_ratio)

    def open_link(self):
        QDesktopServices.openUrl(QUrl(self.video_url))

    def set_thumbnail(self, idx, pixmap):
        if not pixmap.isNull():
            self.lbl_thumb.setPixmap(pixmap)


# ─────────────────────────────────────────────────────────────────────────────
# 4. 메인 UI 클래스
# ─────────────────────────────────────────────────────────────────────────────
class YoutubeUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.workers = []
        self._init_ui()

    def _init_ui(self):
        main_lay = QtWidgets.QVBoxLayout(self)

        tabs = QtWidgets.QTabWidget()
        main_lay.addWidget(tabs)

        self.tab_api = QtWidgets.QWidget()
        self._init_api_tab(self.tab_api)
        tabs.addTab(self.tab_api, "🔍 꿀통 발굴기 (API)")

        tabs.addTab(self._create_link_tab("https://www.channelcrawler.com", "채널 크롤러"), "Channel Crawler")
        tabs.addTab(self._create_link_tab("https://kr.noxinfluencer.com", "녹스 인플루언서"), "Nox Influencer")
        tabs.addTab(self._create_link_tab("https://vling.net", "블링 (Vling)"), "Vling")

    @staticmethod
    def _create_link_tab(url, desc):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setAlignment(QtCore.Qt.AlignCenter)
        lbl = QtWidgets.QLabel(desc);
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 16px; color: #555; margin-bottom: 20px;")
        btn = QtWidgets.QPushButton("사이트 바로가기 🔗")
        btn.setFixedSize(200, 50)
        btn.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #f0f0f0; border-radius: 10px;")
        btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
        lay.addWidget(lbl);
        lay.addWidget(btn)
        return w

    def _init_api_tab(self, parent):
        lay = QtWidgets.QVBoxLayout(parent)

        grp_opt = QtWidgets.QGroupBox("검색 옵션")
        form = QtWidgets.QGridLayout(grp_opt)

        self.line_keyword = QtWidgets.QLineEdit()
        self.line_keyword.setPlaceholderText("검색어 (비우면 카테고리 인기순)")
        self.line_keyword.returnPressed.connect(self.start_search)

        self.combo_cat = QtWidgets.QComboBox()
        self.combo_cat.addItem("전체 (인기 급상승 - 비추천)", None)
        self.combo_cat.addItem("🛠️ 노하우/스타일 (추천)", '26')
        self.combo_cat.addItem("📚 교육", '27')
        self.combo_cat.addItem("🧬 과학/기술", '28')
        self.combo_cat.addItem("🐶 애완동물", '15')
        self.combo_cat.addItem("✈️ 여행/이벤트", '19')
        self.combo_cat.addItem("🎬 인물/블로그", '22')
        self.combo_cat.setCurrentIndex(1)

        self.combo_date = QtWidgets.QComboBox()
        self.combo_date.addItems(["전체 기간", "지난 24시간", "지난 1주", "지난 1달"])
        self.combo_date.setCurrentIndex(0)

        self.combo_duration = QtWidgets.QComboBox()
        self.combo_duration.addItems(["모든 길이", "쇼츠 (4분↓)", "중간 (4~20분)", "롱폼 (20분↑)"])
        self.combo_duration.setCurrentIndex(0)

        self.combo_limit = QtWidgets.QComboBox()
        self.combo_limit.addItems(["30개", "10개", "50개"])
        self.combo_limit.setCurrentIndex(2)  # 50개

        self.spin_sub_max = QtWidgets.QSpinBox()
        self.spin_sub_max.setRange(0, 10000000)
        self.spin_sub_max.setSingleStep(10000)
        self.spin_sub_max.setValue(500000)
        self.spin_sub_max.setSuffix(" 명 이하")
        self.spin_sub_max.setSpecialValueText("제한 없음")

        btn_search = QtWidgets.QPushButton("검색 시작")
        btn_search.clicked.connect(self.start_search)
        btn_search.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px 15px;")

        # 배치
        form.addWidget(QtWidgets.QLabel("카테고리:"), 0, 0)
        form.addWidget(self.combo_cat, 0, 1)
        form.addWidget(QtWidgets.QLabel("키워드:"), 0, 2)
        form.addWidget(self.line_keyword, 0, 3)
        form.addWidget(btn_search, 0, 4, 2, 1)

        form.addWidget(QtWidgets.QLabel("기간:"), 1, 0)
        form.addWidget(self.combo_date, 1, 1)
        form.addWidget(QtWidgets.QLabel("길이:"), 1, 2)
        form.addWidget(self.combo_duration, 1, 3)

        form.addWidget(QtWidgets.QLabel("개수:"), 2, 0)
        form.addWidget(self.combo_limit, 2, 1)
        form.addWidget(QtWidgets.QLabel("구독자:"), 2, 2)
        form.addWidget(self.spin_sub_max, 2, 3)

        lay.addWidget(grp_opt)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.list_widget.setSpacing(5)
        lay.addWidget(self.list_widget)

        self.lbl_status = QtWidgets.QLabel("준비됨 (게임/음악/Live 필터링 적용)")
        self.lbl_status.setStyleSheet("color: #666; font-size: 11px;")
        lay.addWidget(self.lbl_status)

    def start_search(self):
        api_key = os.getenv("YOUTUBE_API_KEY", "")
        if not api_key:
            QtWidgets.QMessageBox.critical(self, "오류", "설정(.env 또는 settings)에 YOUTUBE_API_KEY가 없습니다.")
            return

        keyword = self.line_keyword.text().strip()
        cat_data = self.combo_cat.currentData()

        d_idx = self.combo_date.currentIndex()
        d_map = {0: 'any', 1: 'day', 2: 'week', 3: 'month'}
        date_filter = d_map.get(d_idx, 'any')

        dur_idx = self.combo_duration.currentIndex()
        dur_map = {0: 'any', 1: 'short', 2: 'medium', 3: 'long'}
        duration_filter = dur_map.get(dur_idx, 'any')

        l_text = self.combo_limit.currentText()
        max_results = int(l_text.replace("개", ""))
        max_sub = self.spin_sub_max.value()

        self.list_widget.clear()

        status_msg = f"분석 중... (카테고리: {self.combo_cat.currentText()})"
        self.lbl_status.setText(status_msg)

        self.workers.clear()

        worker = YoutubeWorker(api_key, keyword, max_sub, date_filter, max_results, duration_filter, cat_data)
        worker.finished.connect(self.on_search_finished)
        self.workers.append(worker)
        worker.start()

    def on_search_finished(self, results, msg):
        if not results:
            self.lbl_status.setText("결과 없음")
            QtWidgets.QMessageBox.information(self, "알림", msg)
            return

        self.lbl_status.setText(f"분석 완료: {len(results)}개 발견")

        for i, item in enumerate(results):
            l_item = QtWidgets.QListWidgetItem(self.list_widget)
            l_item.setSizeHint(QtCore.QSize(0, 110))

            w_item = VideoItemWidget(item)
            self.list_widget.setItemWidget(l_item, w_item)

            if item.get('thumb'):
                loader = ImageLoader(i, item['thumb'])
                loader.loaded.connect(w_item.set_thumbnail)
                self.workers.append(loader)
                loader.start()


def create_youtube_widget(parent=None):
    return YoutubeUI(parent)