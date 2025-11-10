import google.generativeai as genai
from google.generativeai.types import RequestOptions
import google.api_core.exceptions as gax
import random
from typing import Any, Optional, TypedDict
import time
import json
import re
from io import BytesIO
from typing import Optional, Tuple
import settings

import requests
from PIL import Image
from xmlrpc import client as xmlrpc_client

class RespOut(TypedDict):
    text: Optional[str]
    blocked: bool
    finish_reason: Optional[str]
    block_reason: Optional[str]
    safety_ratings: Any
    has_parts: bool


def call_gemini(prompt, temperature=0.6, is_json=False, max_retries=5):
    """
    반환:
      - 성공: str(response_text)
      - SAFETY 차단: "SAFETY_BLOCKED"
      - 실패: "API_ERROR"
    """
    model_name = "gemini-2.5-pro"
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        response_mime_type="application/json" if is_json else "text/plain",
        candidate_count=1,
        # 필요시 최대 토큰 제한:
        # max_output_tokens=1024,
        # top_p=0.9,
    )
    request_options = RequestOptions(timeout=300)

    for attempt in range(max_retries):
        try:
            print(
                f"▶ [Gemini] 시도 {attempt + 1}/{max_retries} | model={model_name} | json={is_json} | temp={temperature}",
                flush=True)

            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(
                [{"role": "user", "parts": [{"text": prompt}]}],  # 권장 포맷
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options=request_options,
            )

            ex = _extract_text_from_parts(resp)
            print(f"   ├─ finish_reason={ex['finish_reason']} | blocked={ex['blocked']} | has_parts={ex['has_parts']}")

            if ex["blocked"]:
                print(f"   ├─ SAFETY 차단(block_reason={ex['block_reason']}) → 중단")
                return "SAFETY_BLOCKED"

            # 정상 텍스트
            if ex["text"]:
                print(f"✅ 응답 수신 (길이={len(ex['text'])})")
                return ex["text"]

            # parts 없음/빈 텍스트인데 finish_reason=STOP → JSON 모드일 경우 저온 재프롬프트 1회
            if is_json:
                print("⚠️ 후보는 있으나 텍스트가 비어 있음 → 저온 재프롬프트 시도(1회)")
                fixed = _low_temp_json_reprompt(model_name, prompt, request_options)
                if fixed and fixed.strip():
                    print(f"✅ 저온 재프롬프트 성공 (길이={len(fixed)})")
                    return fixed
                else:
                    print("   └─ 저온 재프롬프트 실패 → 재시도 루프로 이동")
            else:
                print("⚠️ 후보는 있으나 텍스트가 비어 있음 → 재시도 대상")

        except Exception as exc:
            et = type(exc).__name__
            print(f"❌ 예외 발생 [{et}]: {exc}")
            if not _is_retryable_exception(exc):
                print("🚫 재시도 비대상 오류 → 즉시 실패")
                return "API_ERROR"

        # 재시도 전 대기 (마지막 시도는 대기/출력 생략)
        if attempt < max_retries - 1:
            wait = _backoff(attempt)  # 기존 계산 그대로 사용 (지터 포함)
            # print로만 알리지 말고, 실제로 그 시간만큼 정확히 잔다
            sleep_with_exact(wait, label=f"attempt {attempt + 1} → {attempt + 2}")
        else:
            # 마지막 시도: 재시도 없음, 대기/출력 금지
            pass

    print("❌ 최대 재시도 횟수 초과 → 실패 처리")
    return "API_ERROR"


def _extract_text_from_parts(resp: Any) -> RespOut:
    """
    Gemini SDK 응답에서 텍스트/메타 정보를 '안전하게' 추출합니다.
    - response.text 에는 접근하지 않습니다 (SDK에 따라 ValueError 발생).
    - candidates[0].content.parts 를 신뢰해서 문자열을 모읍니다.
    - SAFETY 차단 여부, finish_reason, block_reason, safety_ratings, parts 존재 여부를 함께 반환합니다.
    """
    out: RespOut = {
        "text": None,
        "blocked": False,
        "finish_reason": None,
        "block_reason": None,
        "safety_ratings": None,
        "has_parts": False,
    }

    try:
        # prompt_feedback에서 차단 사유(있다면) 추출
        prompt_feedback = getattr(resp, "prompt_feedback", None)
        if prompt_feedback is not None:
            # SDK 버전에 따라 속성명이 다를 수 있어 둘 다 시도
            out["block_reason"] = (
                getattr(prompt_feedback, "block_reason", None)
                or getattr(prompt_feedback, "block_reason_message", None)
            )

        candidates = getattr(resp, "candidates", []) or []
        if not candidates:
            return out

        c0 = candidates[0]

        # finish_reason (enum or int → str로 정규화)
        finish_reason = getattr(c0, "finish_reason", None)
        out["finish_reason"] = getattr(finish_reason, "name", None) or str(finish_reason) if finish_reason is not None else None

        # SAFETY 차단 여부
        if out["finish_reason"] and str(out["finish_reason"]).upper() == "SAFETY":
            out["blocked"] = True

        # safety_ratings 그대로 보관 (타입 Any)
        out["safety_ratings"] = getattr(c0, "safety_ratings", None)

        # 본문 parts에서 텍스트 조립
        content = getattr(c0, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            out["has_parts"] = True
            pieces: list[str] = []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:   # ← 타입 체커 OK
                    pieces.append(t)
            if pieces:
                joined = "".join(pieces).strip()
                if joined:
                    out["text"] = joined

        return out

    except Exception as exc:
        # 파싱 중 예외는 로깅만 하고, out 기본값 반환
        print(f"⚠️ 응답 파싱 오류(파트 추출): {type(exc).__name__}: {exc}")
        return out


def _low_temp_json_reprompt(model_name: str, base_prompt: str, request_options: RequestOptions) -> str | None:
    """
    is_json=True인데 parts가 비었을 때, 저온·단호한 형식 지시로 1회 재프롬프트.
    """
    try:
        model = genai.GenerativeModel(model_name)
        strict_prompt = (
            "You must return ONLY valid JSON.\n"
            "Do NOT include any explanations, code fences, or markdown.\n"
            "Return a JSON array or object as requested.\n\n"
            f"{base_prompt}"
        )
        resp = model.generate_content(
            [{"role": "user", "parts": [{"text": strict_prompt}]}],
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
                candidate_count=1,
            ),
            request_options=request_options,
        )
        ex = _extract_text_from_parts(resp)
        print(f"   └─ 저온 재프롬프트 결과: finish_reason={ex['finish_reason']} | has_parts={ex['has_parts']}")
        if ex["blocked"]:
            print("   └─ 저온 재프롬프트도 SAFETY 차단")
            return None
        return ex["text"] if ex["text"] else None
    except Exception as exc:
        print(f"   └─ 저온 재프롬프트 예외: {type(exc).__name__}: {exc}")
        return None


def _is_retryable_exception(exc: Exception) -> bool:
    retryable_types = (
        getattr(gax, "DeadlineExceeded", tuple()),
        getattr(gax, "ServiceUnavailable", tuple()),
        getattr(gax, "ResourceExhausted", tuple()),  # 429
        getattr(gax, "InternalServerError", tuple()),
    )
    if isinstance(exc, retryable_types):
        return True
    msg = str(exc).lower()
    retry_keywords = [
        "deadline exceeded", "service unavailable", "temporarily unavailable",
        "connection reset", "connection aborted", "timed out", "timeout",
        "rate limit", "429", "unavailable", "try again"
    ]
    return any(k in msg for k in retry_keywords)

def _backoff(attempt: int, base: float = 1.0, cap: float = 20.0) -> float:
    wait = min(cap, base * (2 ** attempt))
    wait *= (0.5 + random.random())  # 0.5~1.5 지터
    return wait


def sleep_with_exact(seconds: float, label: Optional[str] = None) -> None:
    """monotonic 기준으로 정확히 seconds만큼 대기. 1초 단위 카운트다운 로그."""
    if seconds <= 0:
        return
    start = time.monotonic()
    end = start + seconds
    if label:
        print(f"⏳ {seconds:.1f}초 대기 후 재시도 | {label}", flush=True)
    else:
        print(f"⏳ {seconds:.1f}초 대기 후 재시도", flush=True)

    # 1초 간격 카운트다운(너무 시끄럽다면 주석처리 가능)
    last_print = int(seconds)
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            break
        rem_int = int(remaining)
        if rem_int < last_print:  # 초가 바뀔 때만 찍음
            print(f"   … 남은 대기: {rem_int}s", flush=True)
            last_print = rem_int
        # 100~200ms 단위로 짧게 잔다 (정확도↑)
        time.sleep(min(0.2, remaining))
    # 실제 경과 확인
    elapsed = time.monotonic() - start
    print(f"⏱️ 실제 대기: {elapsed:.2f}s", flush=True)

def stable_diffusion(article: str, filename: str, description: str, slug: str):
    """
    [개선]
    - Gemini 응답: 엄격 JSON + parts 없음 대응 + 실패 시 대체 프롬프트
    - SD 호출: 재시도/백오프/타임아웃 + 응답 검증
    - 진행 로그 및 캡션 후처리
    """
    try:
        print(f"▶ Gemini로 [{filename}] 이미지 프롬프트/캡션 생성 요청...", flush=True)

        style_guideline = _make_style_guideline(filename)
        meta_prompt = _build_meta_prompt(article, description, style_guideline)

        # Gemini 호출 (is_json=True)
        response_text = call_gemini(meta_prompt, temperature=0.6, is_json=True)

        # 프롬프트/캡션 파싱 + 실패 시 대체
        short_prompt: Optional[str] = None
        image_caption: Optional[str] = None

        try:
            if response_text in ("SAFETY_BLOCKED", "API_ERROR") or not response_text:
                raise ValueError(f"Gemini 실패: {response_text}")

            parsed = json.loads(response_text)
            short_prompt = (parsed.get("image_prompt") or "").strip()
            image_caption = (parsed.get("caption") or "").strip()

            # 필수 키 확인
            if not short_prompt or not image_caption:
                raise ValueError("JSON에 image_prompt/caption 누락")

        except Exception as exc:
            print(f"⚠️ AI 프롬프트/캡션 생성 실패: {exc} → 대체 프롬프트 사용", flush=True)
            short_prompt, image_caption = _fallback_prompt(description)

        # 공통 네거티브(텍스트/워터마크 강력 억제 포함)
        negative = (
            "(text, logo, watermark:1.5), (deformed, distorted, disfigured:1.2), poorly drawn, bad anatomy, "
            "blurry, lowres, nsfw, nude, extra fingers, mutated hands"
            "deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, ugly, blurry, (text, watermark, signature, username, logo:1.4), (mutated hands and fingers:1.5), morbid, mutilated, extra limbs"
        )

        # 최종 프롬프트 구성(너무 장황하면 품질이 흔들리므로 핵심만)
        base_quality = "masterpiece, best quality, ultra-detailed, high resolution, 8k, ultra high res, cinematic photo, soft light"
        final_prompt = f"{base_quality}, {short_prompt}"

        # 로그
        print(f"🖼 prompt: {final_prompt}", flush=True)
        print(f"✍️ caption: {image_caption}", flush=True)

        # Stable Diffusion 호출 (768x512가 블로그 썸네일/본문에 무난)
        b = _sd_txt2img(
            prompt=final_prompt,
            negative_prompt=negative,
            width=768 if filename == "scene" else 640,
            height=512,
            steps=28 if filename == "scene" else 24,
            cfg_scale=7.0,
            sampler_index="Euler a",
            seed=None,  # 필요시 고정값 지정
            timeout=200,
            max_retries=3,
        )

        if not b:
            raise RuntimeError("Stable Diffusion에서 이미지를 받지 못했습니다.")

        # 후처리(웹 최적화 WEBP)
        img = Image.open(BytesIO(b)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=85)
        webp_bytes = buf.getvalue()

        image_file = BytesIO(webp_bytes)
        image_file.name = f"{slug}_{filename}.webp"
        image_file.seek(0)

        # 캡션 정리
        safe_caption = _clean_caption(image_caption)

        media = {
            "name": image_file.name,
            "type": "image/webp",
            "caption": safe_caption,
            "description": (description or "").strip(),
            "bits": xmlrpc_client.Binary(image_file.read()),
        }
        return media, safe_caption

    except Exception as e:
        print(f"⚠️ Stable Diffusion 처리 중 예외: {type(e).__name__}: {e}", flush=True)
        return None, None

def _make_style_guideline(filename: str) -> str:
    if filename == "thumb":
        return "- 스타일: 미니멀리즘, 플랫 디자인, 벡터 아트\n- 구성: 주제를 상징하는 아이콘/오브젝트 중심\n- 텍스트/로고 금지"
    elif filename == "scene":
        return "- 스타일: 극사실적, 고품질 사진\n- 구성: 주제의 개념/상황을 은유적으로 묘사\n- 조명/배경: 자연광 또는 cinematic lighting, 배경 심도(depth of field)\n- 텍스트/로고 금지"
    return "- 스타일: 고품질의 상징 사진\n- 텍스트/로고 금지"

def _build_meta_prompt(article: str, description: str, style_guideline: str) -> str:
    # 글 요약은 너무 길면 모델이 빈 parts를 반환하는 경우가 있어 500자 제한
    summary = (article or "")[:500]
    return (
        "[역할] 당신은 추상 개념을 시각화하는 AI 이미지 프롬프트 엔지니어이자 카피라이터입니다.\n"
        "[지시] 아래 '글 요약'과 '주제'를 참고하여 Stable Diffusion용 'image_prompt'와 블로그 'caption'을 생성하세요.\n"
        "[규칙]\n"
        "- 인물보다 주제를 잘 상징하는 사물/풍경/추상 이미지 위주로 자연스럽게 표현\n"
        "- 프롬프트에는 텍스트(글자)나 로고, 워터마크를 포함하지 말 것\n"
        f"[스타일 가이드]\n{style_guideline}\n"
        "[출력 형식]\n"
        "- 반드시 {\"image_prompt\": \"...\", \"caption\": \"...\"} 형식의 순수 JSON로만 응답\n"
        f"[글 요약] {summary}\n"
        f"[주제] {description}\n"
    )

def _fallback_prompt(description: str) -> Tuple[str, str]:
    # 대체 프롬프트(짧고 안전). 텍스트/로고/워터마크 금지 포함.
    desc = (description or "").strip()
    short_prompt = (
        f"{desc}, symbolic representation, photorealistic or abstract scene, "
        "no text, no logo, no watermark, clean background, well-composed"
    )
    caption = _clean_caption(desc or "관련 주제 이미지")
    return short_prompt, caption

def _sd_txt2img(
    prompt: str,
    negative_prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 28,
    cfg_scale: float = 7.0,
    sampler_index: str = "Euler a",
    seed: Optional[int] = None,
    endpoint: str = "http://127.0.0.1:7890/sdapi/v1/txt2img",
    timeout: int = 180,
    max_retries: int = 3,
) -> Optional[bytes]:
    import json, base64, requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # --- endpoint 정규화 ---
    base = endpoint.split("/sdapi/", 1)[0] if "/sdapi/" in endpoint else endpoint
    txt2img_url = f"{base}/sdapi/v1/txt2img"
    models_url  = f"{base}/sdapi/v1/sd-models"

    # --- 로컬 통신 전용 세션: 프록시/환경변수 무시 + keep-alive 해제 ---
    sess = requests.Session()
    sess.trust_env = False                     # ← HTTP(S)_PROXY 등 무시
    sess.proxies = {"http": None, "https": None}
    sess.headers.update({"User-Agent": "AutoBlog/1.21", "Connection": "close"})

    # 네트워크 안정화(재시도)
    retry = Retry(
        total=2, connect=2, read=2,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET","POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)  # 혹시 모를 https 호출 대비

    # --- payload 구성 ---
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "sampler_index": sampler_index,
        "cfg_scale": cfg_scale,
        "override_settings": {
            "sd_model_checkpoint": "xxmix9realistic_v40.safetensors [18ed2b6c48]"
        },
        "override_settings_restore_afterwards": True,
    }
    if seed is not None:
        payload["seed"] = seed

    # --- Preflight: 모델 존재 확인 (실패해도 진행) ---
    try:
        r = sess.get(models_url, timeout=10)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("application/json"):
            desired = payload["override_settings"]["sd_model_checkpoint"]
            models = r.json() or []
            titles = {m.get("title") or "" for m in models}
            hashes = {(m.get("hash") or "").strip() for m in models}
            names  = {m.get("model_name") or "" for m in models}
            bracket_hash = desired.split("[",1)[1].split("]",1)[0].strip() if "[" in desired and "]" in desired else ""
            if not (desired in titles or desired in names or (bracket_hash and bracket_hash in hashes)):
                print(f"⚠️ 지정 모델 미탐지 → override_settings 제거: {desired}", flush=True)
                payload.pop("override_settings", None)
        else:
            print(f"⚠️ SD 모델 목록 조회 실패: HTTP {r.status_code} @ {models_url}", flush=True)
    except Exception as exc:
        print(f"⚠️ Preflight 예외({type(exc).__name__}): {exc} → 그대로 진행", flush=True)

    # --- 요청 루프 ---
    for attempt in range(max_retries):
        try:
            print(
                f"🎨 SD 요청 {attempt+1}/{max_retries} | {width}x{height}, steps={steps}, cfg={cfg_scale}, sampler={sampler_index} → {txt2img_url}",
                flush=True,
            )
            # r = sess.post(txt2img_url, json=payload, timeout=timeout)
            import httpx  # requests 대신 httpx 추천(더 탄탄)
            transport = httpx.HTTPTransport(retries=0)  # 재시도는 우리 코드에서 관리
            with httpx.Client(transport=transport, timeout=timeout, headers={"Connection": "close"}) as cli:
                r = cli.post(txt2img_url, json=payload)

            try:
                r.raise_for_status()
            except requests.HTTPError as http_exc:
                status = r.status_code
                body_preview = r.text[:300].replace("\n", " ")
                print(f"⚠️ SD HTTP {status}: {http_exc} | Body≈ {body_preview}", flush=True)
                if status == 404:
                    print("👉 /sdapi/v1 경로 없음: WebUI를 '--api'로 실행했는지/포트 확인", flush=True)
                if status == 500 and "override_settings" in payload:
                    print("👉 모델 스위치가 원인일 수 있어 override_settings 제거 후 즉시 재시도", flush=True)
                    payload.pop("override_settings", None)
                    continue
                raise

            # JSON 파싱
            try:
                data = r.json()
            except json.JSONDecodeError:
                snippet = r.text[:300].replace("\n", " ")
                raise ValueError(f"JSON 파싱 실패(CT={r.headers.get('content-type')}) Body≈ {snippet}")

            imgs = data.get("images")
            if not imgs:
                raise ValueError("SD 응답에 'images' 키가 없거나 비었습니다.")
            return base64.b64decode(imgs[0])

        except requests.exceptions.SSLError as exc:  # ← 여기서 TLS EOF 잡힘
            print(f"❗ SSLError(로컬 호출에 TLS 관여): {exc}", flush=True)
            print("   → 프록시/보안SW 개입 가능성. 세션은 proxies 무시/Connection: close로 설정됨.", flush=True)
        except requests.exceptions.ProxyError as exc:
            print(f"❗ ProxyError: {exc} (환경변수 프록시 무시 중)", flush=True)
        except Exception as exc:
            print(f"⚠️ SD 예외[{type(exc).__name__}]: {exc}", flush=True)

        if attempt < max_retries - 1:
            wait = _backoff(attempt, base=1.0, cap=10.0)
            _sleep_exact(wait, label=f"SD retry {attempt+1}→{attempt+2}")
        else:
            print("🚫 SD 재시도 소진", flush=True)
            return None

    return None

def _clean_caption(text: str, max_len: int = 140) -> str:
    # 줄바꿈/공백 정리, 따옴표 과다 제거, 길이 제한
    s = re.sub(r"\s+", " ", text or "").strip()
    s = s.strip('"\'')

    if len(s) > max_len:
        s = s[:max_len - 1].rstrip() + "…"
        s = s[:max_len - 3].rstrip() + "..."
    return s

def _sleep_exact(seconds: float, label: Optional[str] = None) -> None:
    if seconds <= 0:
        return
    start = time.monotonic()
    end = start + seconds
    if label:
        print(f"⏳ {seconds:.1f}초 대기 | {label}", flush=True)
    else:
        print(f"⏳ {seconds:.1f}초 대기", flush=True)
    while True:
        now = time.monotonic()
        if now >= end:
            break
        time.sleep(min(0.2, end - now))
    elapsed = time.monotonic() - start
    print(f"⏱️ 실제 대기: {elapsed:.2f}s", flush=True)

def build_images_to_blog(
    article: str,
    filename: str,
    description: str,
    slug: str,
    *,
    workflow_path: str | None = None,
    width: int | None = None,
    height: int = 512,
    steps: int = 28,
):
    import json
    import time
    import requests
    from io import BytesIO
    from pathlib import Path
    from PIL import Image
    from xmlrpc import client as xmlrpc_client

    # 1) 프롬프트/캡션 준비 ----------------------------------------------
    try:
        print(f"▶ [ComfyBlog] Gemini로 [{filename}] 이미지 프롬프트/캡션 생성 요청.", flush=True)

        style_guideline = _make_style_guideline(filename)
        meta_prompt = _build_meta_prompt(article, description, style_guideline)

        response_text = call_gemini(meta_prompt, temperature=0.6, is_json=True)

        try:
            if response_text in ("SAFETY_BLOCKED", "API_ERROR") or not response_text:
                raise ValueError(f"Gemini 실패: {response_text}")

            parsed = json.loads(response_text)
            short_prompt = (parsed.get("image_prompt") or "").strip()
            image_caption = (parsed.get("caption") or "").strip()
            if not short_prompt or not image_caption:
                raise ValueError("JSON에 image_prompt/caption 누락")
        except Exception as exc:
            print(f"⚠️ [ComfyBlog] AI 프롬프트/캡션 생성 실패: {exc} → 대체 프롬프트 사용", flush=True)
            short_prompt, image_caption = _fallback_prompt(description)

        negative = (
            "(text, logo, watermark:1.5), (deformed, distorted, disfigured:1.2), poorly drawn, bad anatomy, "
            "blurry, lowres, nsfw, nude, extra fingers, mutated hands"
        )
        base_quality = "masterpiece, best quality, ultra-detailed, high resolution"
        final_prompt = f"{base_quality}, {short_prompt}".strip()

    except Exception as e_prompt:
        print(f"❌ [ComfyBlog] 프롬프트 준비 중 오류: {type(e_prompt).__name__}: {e_prompt}", flush=True)
        return None, None

    # 2) ComfyUI 워크플로 로드 ----------------------------------------------
    try:
        try:
            import settings  # type: ignore
            comfy_host = getattr(settings, "COMFY_HOST", "http://127.0.0.1:8188")
            jsons_dir = getattr(settings, "JSONS_DIR", None)
        except Exception:
            comfy_host = "http://127.0.0.1:8188"
            jsons_dir = None

        base_url = str(comfy_host).rstrip("/")

        if workflow_path is not None:
            wf_path = Path(workflow_path)
        else:
            if jsons_dir:
                wf_path = Path(jsons_dir) / "nunchaku_qwen_image_swap.json"
            else:
                wf_path = Path("nunchaku_qwen_image_swap.json")

        if not wf_path.exists():
            raise FileNotFoundError(f"ComfyUI 워크플로 파일을 찾을 수 없습니다: {wf_path}")

        with open(wf_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # ★★★ 블로그에서는 안 쓸 노드 끄기 ★★★
        def _disable_blog_nodes(g: dict) -> None:
            # 업스케일, 비디오, 리액터 계열 끄기
            UPSCALE = {
                "ImageUpscaleWithModel",
                "ImageScale",
                "LatentUpscale",
                "ImageResize",
            }
            VIDEO = {
                "VHS_VideoCombine",
                "VHS_VideoSave",
                "VHS_VideoLoader",
                "VHS_VideoPreview",
                "VideoHelperSuite",
            }
            FACE = {
                "ReActorFaceSwap",
                "ReActorLoader",
            }
            for nid, node in g.items():
                ctype = str(node.get("class_type") or "")
                if ctype in UPSCALE or ctype in VIDEO or ctype in FACE:
                    node.setdefault("inputs", {})
                    node["inputs"]["enabled"] = False

        _disable_blog_nodes(graph)

        # 3) 블로그용으로 prompt / size / steps 주입 -------------------------
        def _set_input(g: dict, nid: str, key: str, val):
            g[str(nid)].setdefault("inputs", {})[key] = val

        target_w = width or (768 if filename == "scene" else 640)
        target_h = height

        for node_id, node in graph.items():
            ctype = str(node.get("class_type") or "")
            inputs_map = node.get("inputs", {})

            if ctype.startswith("Empty") and "width" in inputs_map:
                _set_input(graph, node_id, "width", int(target_w))
                _set_input(graph, node_id, "height", int(target_h))

            if ctype.lower().startswith("cliptextencode") or "text" in inputs_map:
                label = str(node.get("label") or "").lower()
                if "neg" in label or "negative" in label:
                    _set_input(graph, node_id, "text", negative)
                else:
                    _set_input(graph, node_id, "text", final_prompt)

            if ctype == "KSampler":
                _set_input(graph, node_id, "steps", int(steps))

        # 4) /prompt 제출 ----------------------------------------------------
        resp = requests.post(base_url + "/prompt", json={"prompt": graph}, timeout=15)
        resp.raise_for_status()
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI가 prompt_id를 반환하지 않았습니다.")

        # 5) /history 에서 이미지 찾기 ---------------------------------------
        t0 = time.time()
        img_bytes: bytes | None = None
        while time.time() - t0 < 300:
            h = requests.get(base_url + f"/history/{prompt_id}", timeout=10)
            if h.status_code == 200:
                hjson = h.json()
                outputs = hjson.get(prompt_id, {}).get("outputs") or {}
                for _, out_val in outputs.items():
                    imgs = out_val.get("images") or []
                    if imgs:
                        # 이미지인 것부터 고른다
                        def _is_image_name(name: str) -> bool:
                            name = name.lower()
                            return name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".webp")

                        chosen = None
                        for cand in imgs:
                            if _is_image_name(cand.get("filename", "")):
                                chosen = cand
                                break
                        if not chosen:
                            # 전부 동영상이면 그냥 마지막 걸로
                            chosen = imgs[-1]

                        fname = chosen.get("filename")
                        subfolder = chosen.get("subfolder") or ""

                        # 먼저 output으로
                        params = {"filename": fname, "type": "output"}
                        if subfolder:
                            params["subfolder"] = subfolder

                        try:
                            view = requests.get(base_url + "/view", params=params, timeout=30)
                            view.raise_for_status()
                            img_bytes = view.content
                        except requests.HTTPError:
                            # 안 되면 temp로
                            params["type"] = "temp"
                            view = requests.get(base_url + "/view", params=params, timeout=30)
                            view.raise_for_status()
                            img_bytes = view.content

                        break
            if img_bytes:
                break
            time.sleep(1.5)

        if not img_bytes:
            raise RuntimeError("ComfyUI에서 이미지를 받지 못했습니다.")

        # 6) 워드프레스용 media 로 변환 (SD와 비슷하게 리사이즈) ------------
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        target_w = 768 if filename == "scene" else 640
        target_h = 512
        resample_lanczos = getattr(
            getattr(Image, "Resampling", Image),  # 1순위: Image.Resampling
            "LANCZOS",  # 2순위: 거기서 LANCZOS
            1,  # 둘 다 없으면 1 (lanczos) 숫자값
        )

        img = img.resize((target_w, target_h), resample_lanczos)

        buf = BytesIO()
        img.save(buf, format="WEBP", quality=85)
        webp_bytes = buf.getvalue()

        image_file = BytesIO(webp_bytes)
        image_file.name = f"{slug}_{filename}.webp"
        image_file.seek(0)

        safe_caption = _clean_caption(image_caption)

        media = {
            "name": image_file.name,
            "type": "image/webp",
            "caption": safe_caption,
            "description": (description or "").strip(),
            "bits": xmlrpc_client.Binary(image_file.read()),
        }
        return media, safe_caption

    except Exception as e_img:
        print(f"⚠️ [ComfyBlog] 이미지 생성/변환 중 예외: {type(e_img).__name__}: {e_img}", flush=True)
        return None, None





