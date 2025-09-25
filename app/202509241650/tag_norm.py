# app/tag_norm.py
from __future__ import annotations
from typing import Iterable, List

# 공식/커뮤니티에서 통용되는 영어 태그만 화이트리스트로 유지
ACE_TAG_WHITELIST = {
    # Style
    "electronic","rock","pop","funk","soul","cyberpunk","acid jazz","em",
    "soft electric drums","melodic",
    # Scene
    "background music for parties","radio broadcasts","workout playlists",
    # Instrument
    "saxophone","jazz","piano","violin","acoustic guitar","electric bass",
    # Tempo / Production
    "110 bpm","fast tempo","slow tempo","loops","fills",
    # Vocal
    "soft female voice","soft male voice","mixed vocals",
    # Basic vocal bundle
    "clean vocals","natural articulation","warm emotional tone",
    "studio reverb light","clear diction","breath control","balanced mixing",
}

# 한글/동의어 → 영문 표준 태그 매핑
KO_EN_TAG_MAP = {
    # Style
    "전자음악":"electronic", "록":"rock", "록 음악":"rock",
    "팝":"pop", "펑크":"funk", "소울":"soul",
    "사이버펑크":"cyberpunk", "애시드 재즈":"acid jazz",
    "이엠":"em", "소프트 전자 드럼":"soft electric drums",
    "멜로딕":"melodic",

    # Scene
    "파티 배경 음악":"background music for parties",
    "라디오 방송":"radio broadcasts",
    "운동용 플레이리스트":"workout playlists",

    # Instrument
    "색소폰":"saxophone","재즈":"jazz","피아노":"piano","바이올린":"violin",
    "어쿠스틱 기타":"acoustic guitar","일렉트릭 베이스":"electric bass",

    # Tempo / Production
    "110 비피엠":"110 bpm","빠른 템포":"fast tempo","느린 템포":"slow tempo",
    "루프":"loops","필":"fills",

    # Vocal
    "여성 보컬":"soft female voice","남성 보컬":"soft male voice","혼성 보컬":"mixed vocals",
    "클린 보컬":"clean vocals","자연스러운 발음":"natural articulation",
    "따뜻한 감정 톤":"warm emotional tone","가벼운 스튜디오 리버브":"studio reverb light",
    "명확한 딕션":"clear diction","호흡 컨트롤":"breath control","밸런스드 믹싱":"balanced mixing",
}

# 영어 동의어 → 표준화
EN_SYNONYM_MAP = {
    "female vocal":"soft female voice",
    "male vocal":"soft male voice",
    "female voice":"soft female voice",
    "male voice":"soft male voice",
    "mixed vocal":"mixed vocals",
}

def _dedup(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = (s or "").strip().lower()
        if not k: continue
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def normalize_tags_to_english(tags: Iterable[str]) -> List[str]:
    """입력(한/영 뒤섞임) → 영문 표준 태그만, 소문자·중복 제거."""
    out = []
    for t in tags:
        if not t: continue
        s = t.strip()
        # 한글 → 영문
        s = KO_EN_TAG_MAP.get(s, s)
        # 영어 동의어 정규화
        s_l = s.lower()
        s = EN_SYNONYM_MAP.get(s_l, s_l)
        # 화이트리스트 필터
        if s in ACE_TAG_WHITELIST:
            out.append(s)
    return _dedup(out)
