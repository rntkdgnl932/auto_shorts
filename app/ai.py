# app/ai.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence, Optional, Dict, Any, List
import os, re, json
from pathlib import Path

# ── settings 유연 임포트 ─────────────────────────────────────────────
# 권장: ai.py의 임포트 블록 치환
try:
    from app import settings as settings
except Exception:
    try:
        import settings as settings
    except Exception:
        settings = None

# (선택) 하위 호환
S = settings


# ── .env 로드 (python-dotenv) ────────────────────────────────────────
try:
    from dotenv import load_dotenv  # pip install python-dotenv
except Exception:
    load_dotenv = None

def _load_env_once():
    """가급적 .env를 자동 로드. 이미 값이 있으면 덮어쓰지 않음."""
    if any(os.getenv(k) for k in ("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_KEY")):
        return
    # 후보 경로들: CWD, settings.BASE_DIR, ai.py 상위, 그 조상 경로들
    candidates: List[Path] = []
    try:
        candidates.append(Path.cwd() / ".env")
    except Exception:
        pass
    if S and getattr(S, "BASE_DIR", None):
        candidates.append(Path(getattr(S, "BASE_DIR")) / ".env")  # type: ignore
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        candidates.append(p / ".env")
    # 중복 제거 + 존재하는 파일만
    seen, targets = set(), []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            targets.append(c)
    if load_dotenv:
        for dotenv_path in targets:
            try:
                load_dotenv(dotenv_path=dotenv_path, override=False)
            except Exception:
                pass
    else:
        # 폴백: 매우 단순 파서 (KEY=VALUE 줄만)
        for dotenv_path in targets:
            try:
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    if not line or line.strip().startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        os.environ.setdefault(k, v)
            except Exception:
                pass

_load_env_once()

# ── OpenAI SDK ───────────────────────────────────────────────────────
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # 설치 전이라면 None으로 두고 에러 메시지에서 안내

# ── Gemini SDK (선택) ────────────────────────────────────────────────
try:
    import google.generativeai as genai
    try:
        from google.api_core.exceptions import GoogleAPIError  # type: ignore
    except Exception:
        GoogleAPIError = Exception  # type: ignore
except Exception:
    genai = None
    GoogleAPIError = Exception  # type: ignore

# OpenAI 예외 (결제/한도 감지용)
try:
    from openai import BadRequestError
except Exception:
    class BadRequestError(Exception):
        pass

Provider = Literal["openai", "gemini"]

# ── 설정 컨테이너 ────────────────────────────────────────────────────
@dataclass
class AIConfig:
    provider: Provider = "openai"

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "gpt-5-mini"

    # (옵션) Gemini
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-pro"

    def resolved(self) -> "AIConfig":
        def pick(*cands):
            for c in cands:
                if c:
                    return c
            return None

        provider = pick(
            getattr(S, "AI_PROVIDER", None) if S else None,
            os.getenv("AI_PROVIDER"),
            self.provider,
        )
        # OpenAI 키: 여러 이름 허용
        openai_api_key = pick(
            self.openai_api_key,
            getattr(S, "OPENAI_API_KEY", None) if S else None,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_APIKEY"),
            os.getenv("OPENAI_KEY"),
        )
        return AIConfig(
            provider = (provider or "openai").lower(),  # type: ignore
            openai_api_key = openai_api_key,
            openai_base_url = pick(
                self.openai_base_url,
                getattr(S, "OPENAI_BASE_URL", None) if S else None,
                os.getenv("OPENAI_BASE_URL"),
            ),
            openai_model = pick(
                self.openai_model,
                getattr(S, "OPENAI_MODEL", None) if S else None,
                os.getenv("OPENAI_MODEL"),
                "gpt-5-mini",
            ),
            gemini_api_key = pick(
                self.gemini_api_key,
                getattr(S, "GEMINI_API_KEY", None) if S else None,
                os.getenv("GEMINI_API_KEY"),
            ),
            gemini_model = pick(
                self.gemini_model,
                getattr(S, "GEMINI_MODEL", None) if S else None,
                os.getenv("GEMINI_MODEL"),
                "gemini-2.5-pro",
            ),
        )

# ── 본체 ─────────────────────────────────────────────────────────────
class AI:
    def __init__(self, cfg: AIConfig | None = None):
        self.cfg = (cfg or AIConfig()).resolved()
        self._openai = None
        self.default_prefer = (self.cfg.provider or 'openai').lower()
        self._gemini_ready = False
        self._init_clients()

        self.default_prefer = os.getenv("AI_PREFER", "openai").lower()  # "openai" / "gemini"
        self.gemini_model = getattr(self.cfg, "gemini_model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self._gemini_configured = False

    def _init_clients(self):
        """
        OpenAI/Gemini 클라이언트 초기화.
        - httpx가 OpenAI SDK 요구 버전보다 낮으면 OpenAI SDK 인스턴스를 '시도'조차 하지 않고
          REST 셤(Shim)으로 즉시 폴백하여 httpx 내부 __del__ 경고를 원천 차단한다.
        - 기존 호출부 호환: res.choices[0].message.content 형태 유지.
        """

        # ---- httpx 호환성 체크(사전) ----
        def _httpx_is_compatible() -> bool:
            try:
                import httpx  # type: ignore
            except Exception:
                return False
            ver = getattr(httpx, "__version__", "0.0.0")
            try:
                parts = [int(x) for x in ver.split(".")[:3]]
                while len(parts) < 3:
                    parts.append(0)
                major, minor, patch = parts
            except Exception:
                return False
            # OpenAI 최신 SDK는 httpx >= 0.27 권장
            if major > 0:
                return True
            return (minor, patch) >= (27, 0)

        # ---- OpenAI REST 셤(Shim) ----
        class _OpenAIShim:
            def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 60.0):
                self.api_key = (api_key or "").strip()
                self.base_url = (base_url or "https://api.openai.com").rstrip("/")
                self.timeout = float(timeout)
                self.chat = self._Chat(self)

            class _Chat:
                def __init__(self, outer: "_OpenAIShim"):
                    self.completions = _OpenAIShim._Completions(outer)

            class _Completions:
                def __init__(self, outer: "_OpenAIShim"):
                    self._outer = outer

                def create(self, **params):
                    from types import SimpleNamespace
                    try:
                        import httpx  # type: ignore
                    except Exception as exc:
                        raise RuntimeError("httpx가 필요합니다.") from exc
                    url = f"{self._outer.base_url}/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self._outer.api_key}",
                        "Content-Type": "application/json",
                    }
                    with httpx.Client(timeout=self._outer.timeout) as client:
                        resp = client.post(url, headers=headers, json=params)
                        resp.raise_for_status()
                        data = resp.json()
                    content = ""
                    try:
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
                    except Exception:
                        content = ""
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                    )

        # ---- OpenAI ----
        self._openai = None
        key = (self.cfg.openai_api_key or "").strip()
        base_url = (self.cfg.openai_base_url or "").strip() or None

        if key:
            use_sdk = _httpx_is_compatible() and ("OpenAI" in globals() and OpenAI is not None)
            if use_sdk:
                try:
                    if base_url:
                        self._openai = OpenAI(api_key=key, base_url=base_url)  # type: ignore[name-defined]
                    else:
                        self._openai = OpenAI(api_key=key)  # type: ignore[name-defined]
                except Exception:
                    # 어떤 이유로든 SDK 생성 실패하면 조용히 셤으로 폴백
                    self._openai = _OpenAIShim(api_key=key, base_url=base_url, timeout=60.0)
            else:
                # 사전 체크에서 불합격: SDK 시도 자체를 안 함 → __del__ 경고 원천 차단
                self._openai = _OpenAIShim(api_key=key, base_url=base_url, timeout=60.0)

        # ---- Gemini ----
        self._gemini_ready = False
        gkey = (self.cfg.gemini_api_key or "").strip()
        if gkey and ("genai" in globals() and genai is not None):
            try:
                genai.configure(api_key=gkey)  # type: ignore[name-defined]
                self._gemini_ready = True
            except Exception:
                self._gemini_ready = False

    # ---------- 내부 공용 호출 ----------
    def _ask_openai(self, system: str, prompt: str, **kwargs) -> str:
        """
        OpenAI 호출 (Chat Completions). 모델별 호환을 위해 temperature는 보내지 않는다.
        response_format 등 필요한 값만 kwargs로 전달 가능.
        """
        if self._openai is None:
            raise RuntimeError("OpenAI client is not initialized")

        params: Dict[str, Any] = {
            "model": getattr(self.cfg, "openai_model", None) or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        rf = kwargs.get("response_format")
        if rf:
            params["response_format"] = rf

        res = self._openai.chat.completions.create(**params)
        return (res.choices[0].message.content or "").strip()

    def _ask_gemini(self, system: str, prompt: str, **kwargs) -> str:
        """
        Gemini 호출. google-generativeai 필요: pip install google-generativeai
        system + user를 하나의 프롬프트로 합쳐 전송.
        """
        api_key = getattr(self.cfg, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:
            raise RuntimeError("google-generativeai 패키지가 필요합니다. `pip install google-generativeai`") from exc

        if not self._gemini_configured:
            genai.configure(api_key=api_key)
            self._gemini_configured = True

        model_name = self.gemini_model or "gemini-2.5-pro"
        model = genai.GenerativeModel(model_name)

        # JSON을 더 잘 내도록 하고 싶으면 mime_type을 application/json으로 바꿔도 됨
        gen_cfg: Dict[str, Any] = {"response_mime_type": "text/plain"}
        if isinstance(kwargs.get("response_format"), dict) and kwargs["response_format"].get("type") == "json_object":
            gen_cfg["response_mime_type"] = "application/json"

        resp = model.generate_content(
            f"{system.strip()}\n\n{prompt.strip()}",
            generation_config=gen_cfg,
        )

        # 표준 추출
        try:
            text = (resp.text or "").strip()
            if text:
                return text
        except Exception:
            pass

        # 후보/파츠 폴백
        try:
            parts: List[str] = []
            for cand in getattr(resp, "candidates", []) or []:
                cont = getattr(cand, "content", None)
                for p in getattr(cont, "parts", []) or []:
                    t = getattr(p, "text", "")
                    if t:
                        parts.append(t)
            return "\n".join(parts).strip()
        except Exception:
            return ""

    def ask_smart(
            self,
            system: str,
            user: str,
            *,
            prefer: str = "openai",
            allow_fallback: bool = True,
            trace=None,
            **kwargs,
    ) -> str:
        """
        공급자 우선순위에 따라 호출하고, 실제 호출한 쪽의 trace 라벨을 남긴다.
        prefer="gemini" 이고 allow_fallback=False 이면 Gemini만 사용.
        """

        def _t(ev: str, msg: str) -> None:
            if trace:
                try:
                    trace(ev, msg)
                except Exception:
                    pass

        order = ["openai", "gemini"] if prefer == "openai" else ["gemini", "openai"]
        last_err: BaseException | None = None

        for provider in order:
            if prefer == "gemini" and not allow_fallback and provider != "gemini":
                continue

            if provider == "openai":
                _t(
                    "openai:request",
                    f"model={getattr(self.cfg, 'openai_model', None) or os.getenv('OPENAI_MODEL', 'gpt-5-mini')}"
                )
                try:
                    # 시그니처에 맞게 포지셔널 전달: (system, prompt)
                    out = self._ask_openai(system, user, **kwargs)
                    _t("openai:success", f"len={len(out)}")
                    return out
                except Exception as e:
                    _t("openai:error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer == "openai" and allow_fallback:
                        continue
                    raise
            else:
                _t("gemini:request", f"model={self.gemini_model}")
                try:
                    # 시그니처에 맞게 포지셔널 전달: (system, prompt)
                    out = self._ask_gemini(system, user, **kwargs)
                    _t("gemini:success", f"len={len(out)}")
                    return out
                except Exception as e:
                    _t("gemini:error", f"{type(e).__name__}: {e}")
                    last_err = e
                    if prefer == "gemini" and allow_fallback:
                        continue
                    raise

        if last_err:
            raise last_err
        raise RuntimeError("ask_smart: no provider attempted")

    def _ask(self, system: str, prompt: str) -> str:
        """
        기본: OpenAI 먼저, 결제/한도 오류면 Gemini 폴백.
        provider가 'gemini'로 설정된 경우 Gemini만 사용.
        """
        prefer = (getattr(self, "default_prefer", None) or getattr(self.cfg, "provider", "openai")).lower()
        if prefer == "gemini":
            return self._ask_gemini(system, prompt)
        try:
            return self._ask_openai(system, prompt)
        except BadRequestError as err:
            msg = str(err).lower()
            # 결제/한도/크레딧 관련 메시지에서만 폴백
            if any(k in msg for k in ["insufficient_quota", "billing", "credit", "payment", "402"]) and getattr(self, "_gemini_ready", False):
                return self._ask_gemini(system, prompt)
            raise

    # ---------- 프롬프트(장면 1문장) ----------
    def scene_prompt_kor(
        self,
        *,
        section: str,
        scene_hint: str,
        characters: Sequence[str],
        tags: Sequence[str] = (),
        effect: Optional[str] = None,
        motion: Optional[str] = None,
    ) -> str:
        has_female = any("female" in (c or "").lower() for c in characters)
        has_male   = any("male"   in (c or "").lower() for c in characters)
        people_txt = (
            "여자 단독" if has_female and not has_male else
            "남자 단독" if has_male and not has_female else
            "남녀 투샷" if (has_female and has_male) else
            "인물 없음"
        )
        # 태그 4~8개만 (실제 사용자 컨텍스트로 프롬프트에 녹임)
        tags_used = [t for t in (tags or []) if t][:8]

        system_rules = (
            "너는 뮤직비디오 한 장면을 묘사하는 프롬프트 생성기다. "
            "문장 한 줄로, '배경/인물/행동'이 자연스럽게 포함되도록 요약해. "
            "가사 원문을 그대로 쓰지 말고, 장면의 시각적 정보를 압축해라. "
            "인물은 {people}이며 **정면 또는 3/4 각도**, **얼굴 프레임 중앙**, **선명한 조명**을 명시해라. "
            "배경/분위기/렌즈/시간대/조명 등 시각 키워드만 사용하고, ‘노래’·‘가사’ 같은 단어는 금지. "
            "세로 영상에 맞는 안정 구도를 권장한다."
        ).format(people=people_txt)

        user_ctx = {
            "section": section,
            "scene_hint": scene_hint,
            "characters": list(characters),
            "tags": list(tags_used),
            "effect": effect,
            "motion": motion,
        }
        prompt_template = (
            "아래 컨텍스트로 1문장 프롬프트를 만들어줘.\n"
            f"[컨텍스트]\n{json.dumps(user_ctx, ensure_ascii=False)}\n"
            "출력은 딱 한 문장 한국어. 예: "
            "“노을 진 창가에서 여자가 카메라를 정면으로 바라보며 고요히 숨을 고르는 장면, "
            "부드러운 보케와 따뜻한 톤, 얼굴 중앙, 3/4 각도”."
        )
        return self._ask(system_rules, prompt_template)

    # ---------- 제목/가사/태그 생성 ----------
    def generate_title_lyrics_tags(
            self,
            *,
            prompt: str,
            duration_min: int,
            title_in: str = "",
            allowed_tags: Sequence[str] = (),
            language: str = "ko",
            duration_sec: int | None = None,
            trace: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # ── 길이 스펙 결정 ──
        # 우선순위: duration_sec 인자 > 프롬프트 내 '20초' 등 힌트 > duration_min*60
        seconds_hint = (
            int(duration_sec) if duration_sec else
            (self._extract_seconds_hint(prompt) or (int(duration_min) * 60 if duration_min else 60))
        )
        # 기존 분(min) 스펙(60/120/180)은 그대로 유지
        dur_spec: Dict[int, Dict[str, str]] = {
            1: dict(
                target="~60s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [bridge] 4–6 lines → [chorus] 6–8 lines\n"
                    "- Chorus twice, total ~24–32 lines"
                ),
            ),
            2: dict(
                target="~120s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [verse] 8–10 lines → [chorus] 6–8 lines\n"
                    "- Chorus twice, total ~32–40 lines"
                ),
            ),
            3: dict(
                target="~180s",
                structure=(
                    "[verse] 8–10 lines → [chorus] 6–8 lines → [verse] 8–10 lines → "
                    "[chorus] 6–8 lines → [bridge] 4–6 lines → [chorus] 6–8 lines → [outro] 2–4 lines\n"
                    "- Chorus three times (last chorus can repeat), total ~44–56 lines"
                ),
            ),
        }
        # seconds_hint에 따라 요약 스펙
        if seconds_hint <= 30:
            spec = dict(target="≤30s", structure="[verse] 2–3 lines → [chorus] 2–3 lines (total 4–6 lines).")
        elif seconds_hint <= 60:
            spec = dict(target="31–60s", structure="[verse] 4–6 lines → [chorus] 4–6 lines (total 8–12 lines).")
        else:
            spec = dur_spec.get(max(1, min(3, int(duration_min or 2))), dur_spec[2])

        allowed_str = ", ".join(sorted({t for t in allowed_tags})) if allowed_tags else ""

        sys_rule = (
            "You are a Korean lyricist and music director. Return ONE JSON object only:\n"
            '{"title":"...", "lyrics":"...", "tags":["...", "..."], "tags_pick":["...", "..."]}\n'
            "- `lyrics` MUST use ONLY these headers: [verse], [chorus], [bridge], [outro].\n"
            f"- Target duration: {spec['target']}. Structure guideline:\n{spec['structure']}\n"
            "- Writing style: concise, singable Korean lines (natural prosody), everyday words.\n"
            "- TAGS MUST BE ENGLISH (ACE-Step style), 4–8 items.\n"
            "- If ALLOWED_TAGS are provided, pick 4–10 items ONLY from them that best match mood/instrumentation "
            "and put them in `tags_pick`.\n"
            "- Do NOT include any extra text outside the JSON."
        )
        if allowed_str:
            sys_rule += f"\nALLOWED_TAGS: {allowed_str}\n"

        user_req = {
            "prompt": prompt,
            "duration_min": duration_min,
            "title_hint": title_in,
            "language": language
        }
        ask = (
            "Generate title, lyrics, and tags for the request below. Output JSON ONLY, no code block.\n\n"
            f"[REQUEST]\n{json.dumps(user_req, ensure_ascii=False)}"
        )

        out = self.ask_smart(sys_rule, ask, prefer=None, trace=trace)
        data = self._safe_json(out)

        # ── 안전 보정 + 형식 정리 ──
        data["title"] = self._enforce_title(data.get("title", ""), prompt)
        raw_lyrics = str(data.get("lyrics", "")).strip()

        # 1) 헤더가 같은 줄에 붙은 경우 분리
        raw_lyrics = self._fix_inline_headers(raw_lyrics)
        # 2) 허용 헤더만 정상화
        raw_lyrics = self._normalize_sections(raw_lyrics)
        # 3) 파싱 → 길이 기반 섹션/줄수 강제 컷
        sections = self._parse_sections(raw_lyrics)
        sections = self._enforce_duration_structure(sections, seconds_hint)
        data["lyrics"] = self._format_sections(sections)

        # 태그 정리
        data["tags"] = self._normalize_tags(data.get("tags"))
        picks_raw = self._normalize_tags(data.get("tags_pick"))
        if allowed_tags:
            allowed_set = {t.lower() for t in allowed_tags}
            picks_raw = [t for t in picks_raw if t.lower() in allowed_set]
        data["tags_pick"] = list(dict.fromkeys(picks_raw))[:12]

        return data

    # ---------- JSON/정규화 유틸 ----------
    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        """
        모델이 코드펜스나 설명을 섞어보내도 JSON만 뽑아 안전 파싱
        """
        t = (text or "").strip()
        # 코드펜스 제거
        if t.startswith("```"):
            t = t.strip("`").strip()
        # JSON 블록만 추출
        s, e = t.find("{"), t.rfind("}")
        if 0 <= s < e:
            frag = t[s:e+1]
            try:
                return json.loads(frag)
            except Exception:
                pass
        # 라스트 찬스
        try:
            return json.loads(t)
        except Exception:
            return {"title": "", "lyrics": "", "tags": [], "tags_pick": []}

    @staticmethod
    def _enforce_title(title: str, fallback_prompt: str) -> str:
        t = (title or "").strip()
        if not t or t in {"무제", "제목", "Untitled", "untitled"}:
            t = (fallback_prompt or "노래").strip()
            t = re.sub(r"[^ㄱ-ㅎ가-힣0-9A-Za-z\s]", "", t)
            t = t.split()[0][:12] if t else "노래"
        return t

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen, ret = set(), []
        for it in items:
            k = (it or "").strip()
            if not k:
                continue
            if k not in seen:
                seen.add(k)
                ret.append(k)
        return ret

    def _normalize_tags(self, tags) -> list[str]:
        if isinstance(tags, str):
            parts = [p.strip() for p in re.split(r"[,\n/;]+", tags) if p.strip()]
        elif isinstance(tags, list):
            parts = [str(p).strip() for p in tags if str(p).strip()]
        else:
            parts = []
        # 영문만
        parts = [p for p in parts if re.search(r"[A-Za-z]", p)]
        parts = self._dedup_keep_order(parts)

        # 5개 미만이면 보강할 기본 성향
        basic = [
            "clean vocals",
            "natural articulation",
            "warm emotional tone",
            "studio reverb light",
            "clear diction",
            "breath control",
            "balanced mixing",
        ]
        if len(parts) < 5:
            parts = self._dedup_keep_order(parts + basic)

        return parts[:12]

    @staticmethod
    def _normalize_sections(text: str) -> str:
        if not text:
            return text
        out_lines, has_tag = [], False
        for ln in text.splitlines():
            stripped = ln.strip()
            # 이미 정식 라벨이면 통과
            if re.match(r"^\[(verse|chorus|bridge|outro)(\s+\d+)?]\s*$", stripped, flags=re.IGNORECASE):
                out_lines.append(stripped.lower())
                has_tag = True
                continue
            # 한국어 라벨을 치환
            m = re.match(r"^\s*\(?\s*(\d+)\s*절\s*\)?\s*[:：)]*\s*$", stripped)
            if m:
                out_lines.append(f"[verse {m.group(1)}]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*후\s*렴\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[chorus]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*브\s*릿\s*지\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[bridge]")
                has_tag = True
                continue
            if re.match(r"^\s*\(?\s*아\s*웃\s*트\s*로\s*\)?\s*[:：)]*\s*$", stripped):
                out_lines.append("[outro]")
                has_tag = True
                continue
            out_lines.append(ln)
        if not has_tag and out_lines:
            out_lines.insert(0, "[verse]")
        return "\n".join(out_lines)

    @staticmethod
    def _extract_seconds_hint(text: str) -> Optional[int]:
        if not text:
            return None
        m = re.search(r"(\d{1,3})\s*(초|s|sec|secs|second|seconds)\b", text, flags=re.I)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _fix_inline_headers(text: str) -> str:
        """한 줄에 헤더와 가사가 붙어버린 경우 분리"""
        if not text:
            return ""
        lines = []
        for ln in text.splitlines():
            ln2 = re.sub(r"(?i)\[(verse|chorus|bridge|outro)]\s*", lambda m: m.group(0)+"\n", ln, count=1)
            parts = [p for p in ln2.splitlines() if p.strip()]
            lines.extend(parts)
        return "\n".join(lines)

    @staticmethod
    def _parse_sections(text: str) -> list[tuple[str, list[str]]]:
        """가사를 섹션 단위로 파싱 → [(section, [lines...])]"""
        cur_sec: Optional[str] = None
        cur_lines: list[str] = []
        ret: list[tuple[str, list[str]]] = []
        for ln in (text or "").splitlines():
            s = ln.strip()
            m = re.match(r"^\[(verse|chorus|bridge|outro)(?:\s+\d+)?]\s*$", s, flags=re.I)
            if m:
                if cur_sec is not None:
                    ret.append((cur_sec, [x for x in cur_lines if x.strip()]))
                cur_sec = m.group(1).lower()
                cur_lines = []
            else:
                if s:
                    cur_lines.append(s)
        if cur_sec is not None:
            ret.append((cur_sec, [ln for ln in cur_lines if ln.strip()]))
        if not ret:
            # 섹션이 하나도 없으면 verse로 묶어 반환
            lines = [ln for ln in (text or "").splitlines() if ln.strip()]
            return [("verse", lines)]
        return ret

    @staticmethod
    def _format_sections(sections: list[tuple[str, list[str]]]) -> str:
        """[(sec, lines)]를 정식 포맷 문자열로."""
        chunks: list[str] = []
        for sec, lines in sections:
            chunks.append(f"[{sec}]")
            chunks.extend(lines)
            chunks.append("")  # 섹션 사이 공백
        return "\n".join(chunks).strip()

    @staticmethod
    def _enforce_duration_structure(
            sections: list[tuple[str, list[str]]], seconds: int
    ) -> list[tuple[str, list[str]]]:
        """
        곡 길이에 맞춰 섹션/행 수를 '컷'한다.
        - ≤30초: [verse] 2–3줄 → [chorus] 2–3줄 (총 4–6줄)
        - 31–60초: [verse] 4–6줄 → [chorus] 4–6줄 (총 8–12줄)
        - 나머지: 그대로 두되 헤더 형식만 정리
        """
        sec = max(1, int(seconds or 0))
        if sec <= 30:
            verse, chorus = None, None
            for s, lines in sections:
                if s == "verse" and verse is None:
                    verse = ("verse", lines[:3])  # 2~3줄 목표, 넉넉히 3으로 컷
                elif s == "chorus" and chorus is None:
                    chorus = ("chorus", lines[:3])
                if verse and chorus:
                    break
            out: list[tuple[str, list[str]]] = []
            if verse:  out.append(verse)
            if chorus: out.append(chorus)
            if not out and sections:
                s0, l0 = sections[0]
                out = [(s0, l0[:8])]
            return out

        if sec <= 60:
            verse, chorus = None, None
            for s, lines in sections:
                if s == "verse" and verse is None:
                    verse = ("verse", lines[:6])  # 4~6줄 목표
                elif s == "chorus" and chorus is None:
                    chorus = ("chorus", lines[:6])
                if verse and chorus:
                    break
            out: list[tuple[str, list[str]]] = []
            if verse:  out.append(verse)
            if chorus: out.append(chorus)
            if not out and sections:
                s0, l0 = sections[0]
                out = [(s0, l0[:8])]
            return out

        # 60초 초과는 구조 가이드만 따르고 그대로 반환
        return sections

    # === [ai.py] 추가 ===
    import os
    import re
    import json

    class AI:
        # ... (기존 코드 유지)

        def segment_lyrics(self, sections: list) -> dict:
            """
            sections = [{"id":"S01","text":"<전체 가사 한 줄>"}]
            반환:
            {
              "segments": [
                {"text": "...의미1...", "reason": "..."},
                {"text": "...의미2...", "reason": "..."},
                ...
              ]
            }
            """
            full = " ".join(((sections[0].get("text") or "").strip()).split())
            if not full:
                return {"segments": []}

            # 1) OpenAI 사용 (환경변수 OPENAI_API_KEY 설정 시)
            if os.getenv("OPENAI_API_KEY"):
                try:
                    # OpenAI responses API (Responses) 사용 예시
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    prompt = (
                        "다음 한국어 가사를 의미 전달 단위(한 구절이 온전한 의미, 임팩트, 행동단위, 묘사단위 등)로만 나눠줘. "
                        "되도록이면 문장이 길지 않게 짧게 나눠줘. "
                        "각 조각은 원문 그대로 보존하고, 불필요한 수정/삭제/치환 금지. "
                        "길이 기준이 아니라 의미 기준. JSON 배열로만 반환: [\"구절1\", \"구절2\", ...]\n\n"
                        f"가사:\n{full}"
                    )
                    res = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt,
                        temperature=0.2,
                    )
                    text = res.output_text.strip()
                    # JSON 배열 파싱 시도
                    arr = json.loads(text)
                    segs = [{"text": t, "reason": "openai-seg"} for t in arr if isinstance(t, str) and t.strip()]
                    return {"segments": segs}
                except Exception:
                    pass

            # 2) Gemini 사용 (환경변수 GOOGLE_API_KEY 설정 시)
            if os.getenv("GOOGLE_API_KEY"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = (
                        "다음 한국어 가사를 의미 전달 단위로만 나눠줘. "
                        "각 조각은 원문 그대로 보존하고, 불필요한 수정/삭제/치환 금지. "
                        "길이 기준이 아니라 의미 기준. JSON 배열만 반환: [\"구절1\", \"구절2\", ...]\n\n"
                        f"가사:\n{full}"
                    )
                    resp = model.generate_content(prompt)
                    text = (resp.text or "").strip()
                    arr = json.loads(text)
                    segs = [{"text": t, "reason": "gemini-seg"} for t in arr if isinstance(t, str) and t.strip()]
                    return {"segments": segs}
                except Exception:
                    pass

            # 3) 둘 다 실패시: 안전한 규칙 기반 백업(하드코딩 문구 금지, 문장부호/접속어 기준 휴리스틱)
            #  - 쉼표/마침표/의문문/접속어(그리고/하지만/혹시 등) 주변으로 나눔
            text = full
            # 문장부호 기준 1차 분리
            parts = re.split(r"[\,\.!\?]\s*", text)
            # 추가 휴리스틱: '혹시', '지금', '같은', '하지만', '그리고' 등 접속/전환 단서 앞에서 자르기
            refined = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                chunks = re.split(r"\s+(?=(혹시|지금|같은|하지만|그리고|그래서|그러면))", p)
                buf = []
                for c in chunks:
                    c = c.strip()
                    if not c:
                        continue
                    if buf:
                        buf.append(c)
                        refined.append(" ".join(buf).strip())
                        buf = []
                    else:
                        buf.append(c)
                if buf:
                    refined.append(" ".join(buf).strip())
            refined = [r for r in refined if r]
            return {"segments": [{"text": r, "reason": "fallback"} for r in refined]}
