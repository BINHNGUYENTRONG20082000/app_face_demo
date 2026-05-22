"""Tham số reconnect / backoff — tách khỏi UI và khỏi InsightFace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StreamConnectionConfig:
    """Cấu hình chỉ cho tầng đọc khung hình (không infer)."""

    # Sau N frame lỗi liên tiếp → đóng và mở lại capture
    read_fails_before_reconnect: int = 4
    # Chờ trước khi thử đọc lại sau từng frame lỗi (trước khi đủ N)
    read_error_sleep_s: float = 0.05
    # Chờ sau khi quyết định reconnect (mất luồng)
    reconnect_delay_s: float = 2.5
    # Backoff khi không mở được nguồn (lần thử tăng dần, có trần)
    open_backoff_base_s: float = 2.5
    open_backoff_cap_s: float = 60.0
    # Xả buffer sau khi mở (ít decode hơn read)
    discard_frames_on_open: int = 6
    # Hàng đợi OpenCV — 2 thường giảm lỗi HEVC/POC so với 1
    cap_buffer_size: int = 2
    # Không có frame tốt trong X giây → đóng/mở lại RTSP (giống mở lại VLC). 0 = chỉ đếm lỗi read.
    rtsp_stale_reconnect_s: float = 20.0
    # Sau Y giây giữ một phiên capture → reconnect chủ động (tránh NVR cắt idle). 0 = tắt.
    rtsp_proactive_reconnect_s: float = 0.0
