# -*- coding: utf-8 -*-
from __future__ import annotations

import hmac
import hashlib
import json
from dataclasses import dataclass
from time import gmtime, strftime
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from app import settings as S

COUPANG_DOMAIN = "https://api-gateway.coupang.com"


def _signed_date_gmt() -> str:
    # 쿠팡 HMAC 예제 포맷: yymmddTHHMMSSZ (GMT+0)
    return strftime("%y%m%d", gmtime()) + "T" + strftime("%H%M%S", gmtime()) + "Z"


def _build_authorization(method: str, url: str, secret_key: str, access_key: str) -> str:
    """
    쿠팡 HMAC 시그니처:
      message = signed-date + method + path + querystring(without '?')
      Authorization = "CEA algorithm=HmacSHA256, access-key=..., signed-date=..., signature=..."
    (쿠팡 공식 가이드/예제와 동일한 형태)
    """
    method = (method or "").upper().strip()

    if not url.startswith("/"):
        raise ValueError(f"url must start with '/': {url}")

    path, *query = url.split("?", 1)
    qs = query[0] if query else ""

    signed_date = _signed_date_gmt()
    msg = f"{signed_date}{method}{path}{qs}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    return (
        "CEA algorithm=HmacSHA256, "
        f"access-key={access_key}, "
        f"signed-date={signed_date}, "
        f"signature={signature}"
    )


@dataclass
class CoupangClient:
    access_key: str
    secret_key: str
    partner_id: str = ""
    sub_id: str = ""

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        timeout: int = 15,
    ) -> Dict[str, Any]:
        method = (method or "").upper().strip()
        path = (path or "").strip()

        if not path.startswith("/"):
            raise ValueError("path must start with '/'")

        qs = ""
        if params:
            # 쿠팡 예제 관례대로 urlencode 사용 (doseq=True로 list 대응)
            qs = urlencode(params, doseq=True)
        url_for_sign = f"{path}?{qs}" if qs else path

        auth = _build_authorization(method, url_for_sign, self.secret_key, self.access_key)

        headers = {
            "Authorization": auth,
            "Content-Type": "application/json",
            # OPEN API 문서에 X-Requested-By가 등장하므로 안전하게 넣습니다.
            # (파트너스에서도 있어도 보통 무해)
            "X-Requested-By": self.partner_id or "shorts_make",
        }

        full_url = COUPANG_DOMAIN + path

        try:
            if method == "GET":
                r = requests.get(full_url, params=params, headers=headers, timeout=timeout)
            elif method == "POST":
                r = requests.post(full_url, params=params, headers=headers, data=json.dumps(body or {}), timeout=timeout)
            else:
                raise ValueError(f"unsupported method: {method}")
        except Exception as e:
            raise RuntimeError(f"[Coupang] request error: {e}")

        if r.status_code >= 400:
            # 쿠팡 에러 응답을 그대로 노출
            raise RuntimeError(f"[Coupang] HTTP {r.status_code}: {r.text[:800]}")

        try:
            return r.json()
        except Exception:
            raise RuntimeError(f"[Coupang] invalid json response: {r.text[:800]}")

    def search_products(self, keyword: str, *, limit: int = 1) -> Dict[str, Any]:
        """
        파트너스 상품 검색:
          GET /v2/providers/affiliate_open_api/apis/openapi/v1/products/search
          params: keyword(required), limit, subId(optional)
        """
        keyword = (keyword or "").strip()
        if not keyword:
            raise ValueError("keyword is empty")

        params: Dict[str, Any] = {"keyword": keyword, "limit": int(limit)}
        if self.sub_id:
            params["subId"] = self.sub_id

        return self._request(
            "GET",
            "/v2/providers/affiliate_open_api/apis/openapi/v1/products/search",
            params=params,
            timeout=20,
        )

    def create_deeplink(self, coupang_url: str) -> Dict[str, Any]:
        """
        딥링크 생성:
          POST /v2/providers/affiliate_open_api/apis/openapi/v1/deeplink
          body: { "coupangUrls": ["..."], "subId": "..."(optional) }
        """
        coupang_url = (coupang_url or "").strip()
        if not coupang_url:
            raise ValueError("coupang_url is empty")

        body: Dict[str, Any] = {"coupangUrls": [coupang_url]}
        if self.sub_id:
            body["subId"] = self.sub_id

        return self._request(
            "POST",
            "/v2/providers/affiliate_open_api/apis/openapi/v1/deeplink",
            body=body,
            timeout=20,
        )


def get_client_from_settings() -> CoupangClient:
    if not S.COUPANG_ACCESS_KEY or not S.COUPANG_SECRET_KEY:
        raise RuntimeError("COUPANG_ACCESS_KEY / COUPANG_SECRET_KEY 가 비어있습니다. (.env 로드 확인)")
    return CoupangClient(
        access_key=S.COUPANG_ACCESS_KEY,
        secret_key=S.COUPANG_SECRET_KEY,
        partner_id=S.COUPANG_PARTNER_ID or "",
        sub_id=S.COUPANG_SUB_ID or "",
    )
