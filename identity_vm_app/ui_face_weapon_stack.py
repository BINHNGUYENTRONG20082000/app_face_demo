"""Chi tiết track: scene lớn + crop mặt + crop từng loại vũ khí (gun, knife, …)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from identity_vm_app.services.video_analyze_media import FACE_THUMB_PX
from identity_vm_app.services.video_match_candidates import (
    candidate_match_percent,
    row_match_candidates,
)

THUMB_PX = FACE_THUMB_PX
SCENE_WIDTH = 300


def render_face_match_candidates(row: Dict[str, Any], *, limit: int = 5) -> None:
    """Top-K tên + % định danh của khung (từ match_candidates_json trong DB)."""
    cands = row_match_candidates(row)[: max(1, int(limit))]
    if not cands:
        st.caption("_Chưa có ứng viên định danh._")
        return
    st.caption("**Top định danh (khung này)**")
    for i, c in enumerate(cands, start=1):
        name = str(c.get("display_name") or "?").strip() or "?"
        pct = candidate_match_percent(c)
        if pct is not None:
            st.markdown(f"{i}. **{name}** — {pct:.1f}%")
        else:
            st.markdown(f"{i}. **{name}**")


def render_track_detail_three_images(
    *,
    show_scene: Callable[[], None],
    show_face: Callable[[], None],
    weapon_shows: List[Tuple[str, Callable[[], None]]],
    armed: bool = False,
    has_scene: bool = True,
    has_face: bool = True,
    face_footer: Optional[Callable[[], None]] = None,
    match_row: Optional[Dict[str, Any]] = None,
) -> None:
    if match_row is not None and face_footer is None:
        face_footer = lambda: render_face_match_candidates(match_row)

    def _face_block() -> None:
        if has_face:
            show_face()
        else:
            st.caption("—")
        if face_footer is not None:
            face_footer()

    st.markdown("**Tổng quan (người + box mặt + box vũ khí)**")
    if has_scene:
        show_scene()
    else:
        st.caption("Chưa có ảnh tổng quan track.")

    n_weapon = len(weapon_shows)
    if n_weapon > 0:
        weights = [1] + [1] * n_weapon
        cols = st.columns(weights)
        with cols[0]:
            st.caption("Crop mặt")
            _face_block()
        for i, (label, show_w) in enumerate(weapon_shows):
            with cols[i + 1]:
                st.caption(f"Crop {label}")
                show_w()
    else:
        c_face, c_w = st.columns(2)
        with c_face:
            st.caption("Crop mặt")
            _face_block()
        with c_w:
            st.caption("Crop vũ khí")
            if armed:
                st.caption("Có vũ khí — chưa có ảnh crop.")
            else:
                st.caption("Không vũ khí")
