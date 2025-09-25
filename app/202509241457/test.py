# test.py
import os
from datetime import date, timedelta
import requests
from collections import defaultdict


def _build_headers() -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    headers = {"Authorization": f"Bearer {api_key}"}
    org = os.getenv("OPENAI_ORG_ID")
    if org:
        headers["OpenAI-Organization"] = org
    project = os.getenv("OPENAI_PROJECT") or os.getenv("OPENAI_PROJECT_ID")
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _get_json(url: str, headers: dict, params: dict) -> dict:
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def check_openai_usage() -> None:
    """
    이번 달 OpenAI 토큰 사용량을 /v1/usage?date=YYYY-MM-DD 로 일자별 조회해 합산.
    - 비용(USD) API는 계정별로 비공개일 수 있으므로 호출하지 않음.
    - .env 권장: OPENAI_PROJECT=prj_xxxxx (프로젝트 키일 때)
    """
    headers = _build_headers()
    base_v1 = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    today = date.today()
    start = today.replace(day=1)
    print(f"기간: {start} ~ {today}")

    # 스모크 테스트: 키/헤더 정상 여부
    try:
        _get_json(f"{base_v1}/models", headers, {})
        print("[OK] /models 접근 성공 (키/헤더 정상)")
    except requests.HTTPError as e:
        text = e.response.text if e.response is not None else ""
        print(f"[ERR] /models 실패: {e} {text}")
        return

    total_in = 0
    total_out = 0
    per_day = []
    per_model = defaultdict(lambda: {"in": 0, "out": 0})

    d = start
    while d <= today:
        params = {"date": d.isoformat()}  # ← 하루 단위 조회
        try:
            j = _get_json(f"{base_v1}/usage", headers, params)
            data = j.get("data", [])
            day_in = 0
            day_out = 0
            for row in data:
                in_toks = int(row.get("n_context_tokens_total", 0))
                out_toks = int(row.get("n_generated_tokens_total", 0))
                model = row.get("snapshot_id") or row.get("model") or "unknown"
                day_in += in_toks
                day_out += out_toks
                per_model[model]["in"] += in_toks
                per_model[model]["out"] += out_toks
            total_in += day_in
            total_out += day_out
            per_day.append((d.isoformat(), day_in, day_out))
            print(f"[OK] {d}: in={day_in} out={day_out}")
        except requests.HTTPError as e:
            text = e.response.text if e.response is not None else ""
            print(f"[ERR] /usage {d}: {e}\n{(text or '').strip()}")
        d += timedelta(days=1)

    print("\n=== 합계(이 달) ===")
    print(f"입력 토큰: {total_in:,}")
    print(f"출력 토큰: {total_out:,}")

    if per_model:
        print("\n[모델별 합계]")
        for model, agg in sorted(per_model.items(), key=lambda x: (-(x[1]['in'] + x[1]['out']), x[0])):
            print(f" - {model}: in={agg['in']:,} out={agg['out']:,}")

    if per_day:
        print("\n[일자별 요약]")
        for day, di, do in per_day:
            print(f" - {day}: in={di:,} out={do:,}")
