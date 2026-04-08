from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from newspaper import Article


def _safe_date_to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _download_image_to_temp(image_url: str) -> str:
    if not image_url:
        return ""
    try:
        response = requests.get(image_url, timeout=15, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()

        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"
        elif "gif" in content_type:
            ext = ".gif"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            return temp_file.name
    except Exception:
        return ""


def _scrape_with_newspaper(url: str) -> Dict[str, Any]:
    article = Article(url)
    article.download()
    article.parse()

    return {
        "title": article.title or "",
        "text": article.text or "",
        "authors": article.authors if article.authors else [],
        "date": _safe_date_to_string(article.publish_date),
        "top_image_url": article.top_image or "",
    }


def _scrape_with_bs4(url: str) -> Dict[str, Any]:
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (TruthLens Scraper)"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join([p for p in paragraphs if p]).strip()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    return {
        "title": title,
        "text": text,
        "authors": [],
        "date": "",
        "top_image_url": "",
    }


def scrape_url(url: str) -> Dict[str, Any]:
    source_domain = urlparse(url).netloc
    result: Dict[str, Any] = {
        "title": "",
        "text": "",
        "image_path": "",
        "date": "",
        "authors": [],
        "source_domain": source_domain,
    }

    if not isinstance(url, str) or not url.strip():
        return {"error": "Could not scrape this URL"}

    top_image_url = ""
    try:
        newspaper_data = _scrape_with_newspaper(url)
        result["title"] = newspaper_data["title"]
        result["text"] = newspaper_data["text"]
        result["authors"] = newspaper_data["authors"]
        result["date"] = newspaper_data["date"]
        top_image_url = newspaper_data["top_image_url"]
    except Exception:
        try:
            bs4_data = _scrape_with_bs4(url)
            result["title"] = bs4_data["title"]
            result["text"] = bs4_data["text"]
            result["authors"] = bs4_data["authors"]
            result["date"] = bs4_data["date"]
            top_image_url = bs4_data["top_image_url"]
        except Exception:
            return {"error": "Could not scrape this URL"}

    image_path = _download_image_to_temp(top_image_url)
    result["image_path"] = image_path

    if not result["title"] and not result["text"] and not result["image_path"]:
        return {"error": "Could not scrape this URL"}

    if not isinstance(result["authors"], list):
        result["authors"] = [str(result["authors"])]
    else:
        result["authors"] = [str(author) for author in result["authors"] if str(author).strip()]

    return result
