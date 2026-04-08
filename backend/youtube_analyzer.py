from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer


def _parse_yt_datetime(value: str) -> datetime:
    # YouTube timestamps are typically ISO with trailing 'Z'.
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _compute_title_similarity_score(titles: List[str]) -> float:
    if len(titles) < 2:
        return 0.0

    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(titles).toarray()
    if matrix.shape[0] < 2:
        return 0.0

    sims = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            v1 = matrix[i]
            v2 = matrix[j]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 == 0.0 or n2 == 0.0:
                sim = 0.0
            else:
                sim = float(np.dot(v1, v2) / (n1 * n2))
            sims.append(sim)

    avg_sim = float(np.mean(sims)) if sims else 0.0
    return max(0.0, min(100.0, avg_sim * 100.0))


def analyze_spread(keywords: str, api_key: str) -> Dict[str, Any]:
    """
    Analyze potential coordinated spread behavior on YouTube for a query.
    """
    if not keywords or not str(keywords).strip():
        return {
            "score": 0.5,
            "coordinated_spread_score": 50,
            "breakdown": {
                "spread_count_score": 50,
                "new_channel_score": 50,
                "title_similarity_score": 50,
                "velocity_score": 50,
            },
            "videos_found": 0,
            "suspicious_channels": [],
        }

    if not api_key or api_key == "your-key-here":
        return {
            "score": 0.5,
            "coordinated_spread_score": 50,
            "breakdown": {
                "spread_count_score": 50,
                "new_channel_score": 50,
                "title_similarity_score": 50,
                "velocity_score": 50,
            },
            "videos_found": 0,
            "suspicious_channels": [],
        }

    try:
        youtube = build("youtube", "v3", developerKey=api_key)

        search_response = (
            youtube.search()
            .list(
                q=keywords,
                part="snippet",
                type="video",
                maxResults=20,
                order="relevance",
            )
            .execute()
        )
        search_items = search_response.get("items", [])
        if not search_items:
            return {
                "score": 0.0,
                "coordinated_spread_score": 0,
                "breakdown": {
                    "spread_count_score": 0,
                    "new_channel_score": 0,
                    "title_similarity_score": 0,
                    "velocity_score": 0,
                },
                "videos_found": 0,
                "suspicious_channels": [],
            }

        video_ids = [
            item.get("id", {}).get("videoId", "")
            for item in search_items
            if item.get("id", {}).get("videoId")
        ]
        channel_ids = list(
            {
                item.get("snippet", {}).get("channelId", "")
                for item in search_items
                if item.get("snippet", {}).get("channelId")
            }
        )

        videos_response = (
            youtube.videos()
            .list(
                part="snippet,statistics",
                id=",".join(video_ids),
                maxResults=20,
            )
            .execute()
        )
        channels_response = (
            youtube.channels()
            .list(
                part="snippet,statistics",
                id=",".join(channel_ids),
                maxResults=50,
            )
            .execute()
        )

        channel_map: Dict[str, Dict[str, Any]] = {}
        for ch in channels_response.get("items", []):
            cid = ch.get("id", "")
            ch_snip = ch.get("snippet", {})
            ch_stats = ch.get("statistics", {})
            channel_map[cid] = {
                "channel_name": ch_snip.get("title", ""),
                "created_at": ch_snip.get("publishedAt", ""),
                "subscriber_count": _safe_int(ch_stats.get("subscriberCount", 0)),
            }

        now = datetime.now(timezone.utc)
        video_rows: List[Dict[str, Any]] = []
        for v in videos_response.get("items", []):
            snip = v.get("snippet", {})
            stats = v.get("statistics", {})
            channel_id = snip.get("channelId", "")
            channel_info = channel_map.get(channel_id, {})

            publish_at = snip.get("publishedAt", "")
            channel_created_at = channel_info.get("created_at", "")
            if not publish_at:
                continue

            publish_dt = _parse_yt_datetime(publish_at)
            channel_created_dt = (
                _parse_yt_datetime(channel_created_at)
                if channel_created_at
                else now
            )

            row = {
                "video_id": v.get("id", ""),
                "title": snip.get("title", ""),
                "channel_name": channel_info.get(
                    "channel_name", snip.get("channelTitle", "")
                ),
                "publish_dt": publish_dt,
                "channel_created_dt": channel_created_dt,
                "view_count": _safe_int(stats.get("viewCount", 0)),
                "like_count": _safe_int(stats.get("likeCount", 0)),
                "subscriber_count": _safe_int(channel_info.get("subscriber_count", 0)),
            }
            video_rows.append(row)

        if not video_rows:
            return {
                "score": 0.0,
                "coordinated_spread_score": 0,
                "breakdown": {
                    "spread_count_score": 0,
                    "new_channel_score": 0,
                    "title_similarity_score": 0,
                    "velocity_score": 0,
                },
                "videos_found": 0,
                "suspicious_channels": [],
            }

        # a) spread_count_score
        earliest = min(r["publish_dt"] for r in video_rows)
        within_48h = sum(
            1
            for r in video_rows
            if (r["publish_dt"] - earliest).total_seconds() <= (48 * 3600)
        )
        spread_count_score = min(100.0, (within_48h / max(1, len(video_rows))) * 100.0)

        # b) new_channel_score
        new_channel_ratios = []
        for r in video_rows:
            channel_age_days = max(
                1.0, (now - r["channel_created_dt"]).total_seconds() / 86400.0
            )
            view_count = max(1, r["view_count"])
            ratio = channel_age_days / view_count
            new_channel_ratios.append(ratio)
        avg_ratio = float(np.mean(new_channel_ratios)) if new_channel_ratios else 1.0
        # Lower ratio => newer channel with high views => more suspicious.
        new_channel_score = min(100.0, (1.0 / (avg_ratio + 1e-9)) * 10.0)

        # c) title_similarity_score
        titles = [r["title"] for r in video_rows]
        title_similarity_score = _compute_title_similarity_score(titles)

        # d) velocity_score
        velocities = []
        for r in video_rows:
            hours_since_publish = max(
                1.0, (now - r["publish_dt"]).total_seconds() / 3600.0
            )
            velocity = r["view_count"] / hours_since_publish
            velocities.append(velocity)
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        # Saturating mapping to 0-100.
        velocity_score = min(100.0, (mean_velocity / 5000.0) * 100.0)

        final_score = (
            0.30 * spread_count_score
            + 0.25 * new_channel_score
            + 0.25 * title_similarity_score
            + 0.20 * velocity_score
        )
        final_score = max(0.0, min(100.0, final_score))

        suspicious_channels = []
        for r, vel in zip(video_rows, velocities):
            channel_age_days = max(
                1.0, (now - r["channel_created_dt"]).total_seconds() / 86400.0
            )
            # Heuristic: very new channel with high velocity.
            if channel_age_days < 180 and vel > 500:
                suspicious_channels.append(r["channel_name"])
        suspicious_channels = sorted(set([c for c in suspicious_channels if c]))[:10]

        breakdown = {
            "spread_count_score": round(spread_count_score, 2),
            "new_channel_score": round(new_channel_score, 2),
            "title_similarity_score": round(title_similarity_score, 2),
            "velocity_score": round(velocity_score, 2),
        }

        return {
            "score": round(final_score / 100.0, 4),
            "coordinated_spread_score": int(round(final_score)),
            "breakdown": breakdown,
            "videos_found": len(video_rows),
            "suspicious_channels": suspicious_channels,
        }
    except Exception:
        return {
            "score": 0.5,
            "coordinated_spread_score": 50,
            "breakdown": {
                "spread_count_score": 50,
                "new_channel_score": 50,
                "title_similarity_score": 50,
                "velocity_score": 50,
            },
            "videos_found": 0,
            "suspicious_channels": [],
        }
