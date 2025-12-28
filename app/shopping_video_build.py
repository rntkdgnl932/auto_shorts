# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.utils import AI


def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name).strip()
    if len(name) > 120:
        name = name[:120].rstrip("_")
    return name or "untitled"


def _contains_hangul(s: str) -> bool:
    # 한글 음절/자모 포함 여부
    return bool(re.search(r"[가-힣ㄱ-ㅎㅏ-ㅣ]", s or ""))


def _ensure_dirs(product_dir: Path) -> Dict[str, Path]:
    imgs = product_dir / "imgs"
    clips = product_dir / "clips"
    imgs.mkdir(parents=True, exist_ok=True)
    clips.mkdir(parents=True, exist_ok=True)
    return {"imgs": imgs, "clips": clips}


@dataclass
class BuildOptions:
    scene_count: int = 6
    style: str = "news_hook"   # news_hook / daily / meme
    hook_level: int = 3        # 1~5
    fps: int = 16
    # AI 실패 시 룰 기반 폴백을 허용할지
    allow_fallback_rule: bool = True


@dataclass
class VideoShoppingBuildInput:
    product_name: str
    description: str
    price: str = ""
    product_url: str = ""
    affiliate_url: str = ""
    product_dir: str = ""  # C:\my_games\shorts_make\products\{상품명}


class ShoppingShortsPromptFactory:
    """
    Gemini에게 'meta + scenes[]'만 JSON으로 받기 위한 프롬프트.
    - ComfyUI 사용 전제: prompt_*는 반드시 영어로만
    """
    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are a senior short-form vertical video director and performance marketer.\n"
            "You MUST return ONLY valid JSON (no markdown, no code fences, no commentary).\n"
            "IMPORTANT:\n"
            "- The field 'prompt' MAY be Korean (this is for human narration/notes).\n"
            "- The fields 'prompt_img', 'prompt_movie', 'prompt_negative' MUST be English ONLY.\n"
            "Focus on hook -> tension -> product as solution -> practical use -> CTA.\n"
            "No medical/guaranteed claims. Avoid exaggerated efficacy.\n"
        )

    @staticmethod
    def build_user_prompt(inp: VideoShoppingBuildInput, options: BuildOptions) -> str:
        # “요약/설명”이 한국어여도, 출력은 영어로 강제.
        sc_min, sc_max = 4, 8
        scene_count = int(options.scene_count or 6)
        if scene_count < sc_min:
            scene_count = sc_min
        if scene_count > sc_max:
            scene_count = sc_max

        hook_level = max(1, min(int(options.hook_level or 3), 5))
        style = (options.style or "news_hook").strip()

        return f"""
[PRODUCT INPUT]
- Product name: {inp.product_name}
- Price: {inp.price}
- Product URL: {inp.product_url}
- Affiliate URL: {inp.affiliate_url}
- Summary/Description (source material, may be Korean):
{inp.description}

[YOUR TASK]
Create a shopping short (9:16) plan.
You must decide:
- Hook intensity and style
- Number of scenes and flow
- The story arc (problem -> implication -> solution -> use -> emotional close/CTA)

[CONSTRAINTS]
1) Output JSON ONLY with keys: meta, scenes
2) Scene count: {scene_count} (you can output between {sc_min} and {sc_max}, but prefer {scene_count})
3) Each scene MUST include:
   - id: 3-digit string like "001"
   - seconds: integer 2~8
   - prompt: Korean narration/scene intention (Korean allowed)
   - prompt_img: English ComfyUI image prompt (must include "no text, no watermark")
   - prompt_movie: English image-to-video prompt (camera motion, angle, action)
   - prompt_negative: English negative prompt (no text, no watermark, no logo, no blur, etc.)
4) Language rule:
   - 'prompt' may be Korean.
   - 'prompt_img', 'prompt_movie', 'prompt_negative' MUST be ENGLISH ONLY. Do NOT use Korean there.
5) No on-screen text. Avoid brand logos. No watermarks.
6) No guaranteed efficacy or absolute claims.

[OUTPUT JSON SCHEMA]
{{
  "meta": {{
    "title": "...",
    "hook_style": "...",
    "target": "...",
    "tone": "...",
    "cta": "..."
  }},
  "scenes": [
    {{
      "id": "001",
      "seconds": 3,
      "prompt": "...",
      "prompt_img": "...",
      "prompt_movie": "...",
      "prompt_negative": "..."
    }}
  ]
}}
""".strip()


def _validate_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Gemini payload is not a JSON object.")
    meta = data.get("meta")
    scenes = data.get("scenes")
    if not isinstance(meta, dict):
        raise ValueError("Gemini payload missing meta object.")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Gemini payload scenes must be a non-empty list.")

    ok_scenes: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes, start=1):
        if not isinstance(sc, dict):
            continue
        sid = str(sc.get("id") or f"{i:03d}")
        if not re.fullmatch(r"\d{3}", sid):
            sid = f"{i:03d}"
        seconds = sc.get("seconds")
        try:
            seconds_i = int(seconds)
        except Exception:
            seconds_i = 3
        seconds_i = max(2, min(seconds_i, 8))

        def pick(k: str) -> str:
            v = sc.get(k)
            return (v if isinstance(v, str) else "").strip()

        prompt = pick("prompt")
        prompt_img = pick("prompt_img")
        prompt_movie = pick("prompt_movie")
        prompt_negative = pick("prompt_negative")

        if not prompt_img:
            prompt_img = f"cinematic product scene, {sid}, no text, no watermark"
        if "no text" not in prompt_img.lower():
            prompt_img = (prompt_img + ", no text, no watermark").strip()

        if not prompt_negative:
            prompt_negative = "text, watermark, logo, lowres, blurry, bad anatomy, extra fingers"

        ok_scenes.append({
            "id": sid,
            "seconds": seconds_i,
            "prompt": prompt,
            "prompt_img": prompt_img,
            "prompt_movie": prompt_movie,
            "prompt_negative": prompt_negative,
        })

    data["meta"] = meta
    data["scenes"] = ok_scenes
    return data


class ShoppingVideoJsonRuleBuilder:
    """
    AI 없이도 동작 가능한 폴백(룰 기반).
    - 영어 프롬프트로만 뽑아준다 (ComfyUI 대응)
    """
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def build(self, product_dir: str | Path, product_data: Dict[str, Any], options: BuildOptions) -> Path:
        pdir = Path(product_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        dirs = _ensure_dirs(pdir)

        product_name = (product_data.get("product_name") or pdir.name).strip()
        desc = (product_data.get("description") or "").strip()

        sc = max(4, min(int(options.scene_count or 6), 8))
        hook_level = max(1, min(int(options.hook_level or 3), 5))

        hook_openers = {
            1: "Have you ever thought about this?",
            2: "This is more common than you think.",
            3: "If this happened at home right now… what would you do?",
            4: "One small moment can turn into a real problem—fast.",
            5: "Three seconds. If you don’t have this covered, you’re exposed.",
        }
        hook_line = hook_openers.get(hook_level, hook_openers[3])

        core = desc.strip()
        if len(core) > 160:
            core = core[:160].rstrip() + "..."

        base = [
            f"{hook_line} {core}",
            f"Show a tense everyday risk scenario inspired by: {core}",
            f"Introduce the product as a calm, practical solution: {product_name}",
            f"Show how it’s used, focusing on realism and convenience (no exaggerated claims).",
            f"Close with a reassuring emotion: preparedness and peace of mind.",
            f"CTA: If you want to be ready, check the link and choose a setup that fits your space.",
        ]
        base = base[:sc]

        scenes: List[Dict[str, Any]] = []
        for i in range(1, sc + 1):
            sid = f"{i:03d}"
            seconds = 3 if i < sc else 4
            prompt = base[i - 1]

            prompt_img = (
                f"vertical 9:16, cinematic realistic scene, {prompt}, "
                f"product-focused composition, natural lighting, no text, no watermark"
            )
            prompt_movie = (
                f"vertical 9:16, subtle camera motion, slow push-in, handheld realism, "
                f"match the scene: {prompt}"
            )
            prompt_negative = "text, watermark, logo, brand mark, lowres, blurry, bad anatomy, extra fingers"

            scenes.append({
                "id": sid,
                "seconds": seconds,
                "prompt": prompt,
                "prompt_img": prompt_img,
                "prompt_movie": prompt_movie,
                "prompt_negative": prompt_negative,
                "img_file": str(dirs["imgs"] / f"{sid}.PNG"),
                "movie_file": str(dirs["clips"] / f"{sid}.mp4"),
            })

        out = {
            "schema": "shopping_shorts_v1",
            "style": "shopping_short",
            "product": {
                "product_name": product_name,
                "safe_name": _sanitize_name(product_name),
                "price": (product_data.get("price") or "").strip(),
                "product_url": (product_data.get("product_url") or "").strip(),
                "affiliate_url": (product_data.get("affiliate_url") or "").strip(),
                "product_dir": str(pdir),
            },
            "summary_source": desc,
            "meta": {
                "title": f"{product_name} | Be ready, not surprised",
                "hook_style": options.style,
                "target": "general shoppers",
                "tone": "tense-to-relief",
                "cta": "Check the link and set up your own readiness.",
            },
            "defaults": {
                "image": {"width": 720, "height": 1280},
                "movie": {"fps": int(options.fps)},
            },
            "audit": {
                "generated_at": _now_str(),
                "source": "rule_fallback",
                "note": "Scenes only. English prompts for ComfyUI.",
            },
            "scenes": scenes,
        }

        out_path = pdir / "video_shopping.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        self.on_progress(f"[video] (rule) created: {out_path}")
        return out_path


class ShoppingVideoJsonBuilder:
    """
    메인 엔트리(이 이름만 import해서 쓰면 됨)
    - 기본: Gemini로 meta + scenes 생성
    - 파일 규칙: img_file/movie_file 주입
    - prompt_*는 영어 강제 (한글 감지 시 1회 재요청)
    - 옵션으로 룰 폴백 가능
    """
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None, ai: Optional[AI] = None):
        self.on_progress = on_progress or (lambda msg: None)
        self.ai = ai or AI()

    def build(self, product_dir: str | Path, product_data: Dict[str, Any], options: Optional[BuildOptions] = None) -> Path:
        options = options or BuildOptions()

        pdir = Path(product_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        dirs = _ensure_dirs(pdir)

        product_name = (product_data.get("product_name") or pdir.name).strip()
        desc = (product_data.get("description") or "").strip()

        if not product_name:
            raise ValueError("product_name is empty.")
        if not desc:
            raise ValueError("description(요약/설명) is empty.")

        inp = VideoShoppingBuildInput(
            product_name=product_name,
            description=desc,
            price=(product_data.get("price") or "").strip(),
            product_url=(product_data.get("product_url") or "").strip(),
            affiliate_url=(product_data.get("affiliate_url") or "").strip(),
            product_dir=str(pdir),
        )

        system = ShoppingShortsPromptFactory.build_system_prompt()
        user = ShoppingShortsPromptFactory.build_user_prompt(inp, options)

        self.on_progress("[video] Gemini request: building meta + scenes ...")
        try:
            raw = self.ai.ask_smart(
                system,
                user,
                prefer="gemini",
                allow_fallback=False,
                response_format={"type": "json_object"},
            )
            data = json.loads(raw)
            data = _validate_payload(data)

            # 영어 강제: 한글 감지 시 1회 '영어로만 정리' 재요청
            if self._has_any_korean_in_prompts(data):
                self.on_progress("⚠ Korean detected in prompts. Retrying once to force English only...")
                data = self._retry_force_english(data)

            scenes: List[Dict[str, Any]] = data["scenes"]
            for i, sc in enumerate(scenes, start=1):
                sid = str(sc.get("id") or f"{i:03d}")
                if not re.fullmatch(r"\d{3}", sid):
                    sid = f"{i:03d}"
                    sc["id"] = sid
                sc["img_file"] = str(dirs["imgs"] / f"{sid}.PNG")
                sc["movie_file"] = str(dirs["clips"] / f"{sid}.mp4")

            out = {
                "schema": "shopping_shorts_v1",
                "style": "shopping_short",
                "product": {
                    "product_name": product_name,
                    "safe_name": _sanitize_name(product_name),
                    "price": inp.price,
                    "product_url": inp.product_url,
                    "affiliate_url": inp.affiliate_url,
                    "product_dir": str(pdir),
                },
                "summary_source": desc,
                "meta": data["meta"],
                "defaults": {
                    "image": {"width": 720, "height": 1280},
                    "movie": {"fps": int(options.fps)},
                },
                "audit": {
                    "generated_at": _now_str(),
                    "source": "gemini",
                    "note": "Scenes only. English prompts for ComfyUI.",
                },
                "scenes": scenes,
            }

            out_path = pdir / "video_shopping.json"
            out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
            self.on_progress(f"[video] done: {out_path}")
            return out_path

        except Exception as e:
            self.on_progress(f"❌ Gemini build failed: {e}")

            if options.allow_fallback_rule:
                self.on_progress("[video] fallback to rule-based builder...")
                rule = ShoppingVideoJsonRuleBuilder(on_progress=self.on_progress)
                return rule.build(pdir, product_data, options)

            raise

    def _has_any_korean_in_prompts(self, data: Dict[str, Any]) -> bool:
        scenes = data.get("scenes") or []
        if not isinstance(scenes, list):
            return False
        for sc in scenes:
            if not isinstance(sc, dict):
                continue
            for k in ("prompt_img", "prompt_movie", "prompt_negative"):
                if _contains_hangul(str(sc.get(k) or "")):
                    return True
        return False

    def _retry_force_english(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 구조 그대로 두고, 모든 prompt 필드 영어로만 번역/정리 요청
        system = (
            "You are a strict translator and short-form prompt editor.\n"
            "Return ONLY valid JSON. No markdown.\n"
            "Convert ALL text fields to natural English.\n"
            "Do NOT add new keys. Keep meta + scenes structure.\n"
        )
        user = (
            "Translate and rewrite the following JSON so that ALL strings are English ONLY.\n"
            "Keep keys and ids exactly. Ensure prompt_img contains 'no text, no watermark'.\n"
            "JSON:\n"
            + json.dumps(data, ensure_ascii=False)
        )
        raw = self.ai.ask_smart(
            system,
            user,
            prefer="gemini",
            allow_fallback=False,
            response_format={"type": "json_object"},
        )
        fixed = json.loads(raw)
        fixed = _validate_payload(fixed)
        return fixed


class ShoppingImageGenerator:
    """
    ComfyUI 연동은 추후 붙이기(스켈레톤).
    """
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_images(self, video_json_path: str | Path, skip_if_exists: bool = True) -> None:
        vpath = Path(video_json_path)
        data = json.loads(vpath.read_text(encoding="utf-8"))
        scenes = data.get("scenes") or []
        if not isinstance(scenes, list):
            raise ValueError("video_shopping.json scenes must be a list.")

        for sc in scenes:
            img = Path(sc.get("img_file") or "")
            sid = sc.get("id") or img.stem
            if not str(img):
                continue
            if skip_if_exists and img.exists():
                self.on_progress(f"[img] SKIP exists: {sid}")
                continue
            self.on_progress(f"[img] (TODO) request: {sid} -> {img}")
        self.on_progress("[img] done (skeleton).")


class ShoppingMovieGenerator:
    """
    I2V 연동은 추후 붙이기(스켈레톤).
    """
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def generate_movies(self, video_json_path: str | Path, skip_if_exists: bool = True, fps: int = 16) -> None:
        vpath = Path(video_json_path)
        data = json.loads(vpath.read_text(encoding="utf-8"))
        scenes = data.get("scenes") or []
        if not isinstance(scenes, list):
            raise ValueError("video_shopping.json scenes must be a list.")

        for sc in scenes:
            mv = Path(sc.get("movie_file") or "")
            sid = sc.get("id") or mv.stem
            if not str(mv):
                continue
            if skip_if_exists and mv.exists():
                self.on_progress(f"[mov] SKIP exists: {sid}")
                continue
            self.on_progress(f"[mov] (TODO) request: {sid} -> {mv} (fps={fps})")
        self.on_progress("[mov] done (skeleton).")


class ShoppingShortsPipeline:
    """
    상위 오케스트레이터
    """
    def __init__(self, on_progress: Optional[Callable[[str], None]] = None):
        self.on_progress = on_progress or (lambda msg: None)

    def run_all(
        self,
        product_dir: str | Path,
        product_data: Dict[str, Any],
        options: Optional[BuildOptions] = None,
        build_json: bool = True,
        build_images: bool = False,
        build_movies: bool = False,
        skip_if_exists: bool = True,
    ) -> Path:
        options = options or BuildOptions()

        self.on_progress("[pipe] start")
        builder = ShoppingVideoJsonBuilder(on_progress=self.on_progress)
        vpath = Path(product_dir) / "video_shopping.json"

        if build_json:
            vpath = builder.build(product_dir=product_dir, product_data=product_data, options=options)

        if build_images:
            img_gen = ShoppingImageGenerator(on_progress=self.on_progress)
            img_gen.generate_images(vpath, skip_if_exists=skip_if_exists)

        if build_movies:
            mov_gen = ShoppingMovieGenerator(on_progress=self.on_progress)
            mov_gen.generate_movies(vpath, skip_if_exists=skip_if_exists, fps=int(options.fps))

        self.on_progress("[pipe] end")
        return vpath
