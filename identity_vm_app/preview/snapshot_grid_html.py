"""HTML lưới camera — polling snapshot (mỗi ô một camera, không dùng N luồng MJPEG)."""

from __future__ import annotations

import html
import json
from typing import List


def build_snapshot_grid_html(
    api_base: str,
    camera_ids: List[str],
    n_cols: int,
    *,
    poll_fps: float = 8.0,
) -> tuple[str, int]:
    if not camera_ids:
        return "<p>Không có camera.</p>", 80
    cols = max(1, min(6, int(n_cols)))
    n = len(camera_ids)
    api_js = json.dumps(api_base.rstrip("/"))
    cams_js = json.dumps([str(cid) for cid in camera_ids])
    n_rows = (n + cols - 1) // cols
    row_h = 300
    height_px = min(2400, 48 + n_rows * row_h)

    fps = max(2.0, min(15.0, float(poll_fps)))
    per_cam_ms = max(180, int(1000.0 / fps))
    max_inflight = min(n, max(4, min(12, int(fps * 1.25) + 3)))
    stagger_ms = max(10, per_cam_ms // max(1, n))

    cells: list[str] = []
    for i, cid in enumerate(camera_ids):
        esc = html.escape(str(cid))
        cells.append(
            f'<div class="cell" data-idx="{i}">'
            f'<div class="lbl" id="lbl-{i}">{esc}</div>'
            f'<img data-cid="{esc}" data-idx="{i}" alt="{esc}" decoding="async" />'
            f"</div>"
        )
    grid_html = "".join(cells)

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
  body {{ margin:0; background:#0f172a; font-family:system-ui,sans-serif; }}
  .grid {{ display:grid; grid-template-columns:repeat({cols},1fr); gap:8px; padding:6px; }}
  .cell {{ background:#111827; border-radius:8px; padding:6px; min-height:{row_h - 40}px; }}
  .lbl {{ font-size:12px; color:#9ca3af; margin-bottom:4px; }}
  .cell img {{
    width:100%; max-height:260px; min-height:120px; object-fit:contain;
    background:#1f2937; border-radius:4px; display:block;
  }}
</style></head><body>
<div class="grid">{grid_html}</div>
<script>
(function() {{
  const API = {api_js};
  const CAMS = {cams_js};
  const MAX_INFLIGHT = {max_inflight};
  const PER_CAM_MS = {per_cam_ms};
  const STAGGER_MS = {stagger_ms};
  let inflight = 0;
  const queue = [];

  function imgFor(cid) {{
    return document.querySelector('img[data-cid="' + cid.replace(/"/g, "") + '"]');
  }}

  function drainQueue() {{
    while (inflight < MAX_INFLIGHT && queue.length) {{
      const job = queue.shift();
      job();
    }}
  }}

  function swapWhenReady(img, blobUrl) {{
    return new Promise((resolve, reject) => {{
      const pre = new Image();
      pre.decoding = "async";
      pre.onload = () => {{
        if (img._u) URL.revokeObjectURL(img._u);
        img._u = blobUrl;
        img.src = blobUrl;
        resolve();
      }};
      pre.onerror = () => {{
        URL.revokeObjectURL(blobUrl);
        reject();
      }};
      pre.src = blobUrl;
    }});
  }}

  function refreshOne(cid) {{
    const img = imgFor(cid);
    if (!img) return;
    const run = () => {{
      inflight++;
      const ac = new AbortController();
      const t = setTimeout(() => ac.abort(), 12000);
      const url = API + "/ivm/preview/" + encodeURIComponent(cid)
        + "/snapshot.jpg?wait_s=0&t=" + Date.now();
      fetch(url, {{ signal: ac.signal, cache: "no-store" }})
        .then(r => {{
          if (!r.ok) throw new Error("http");
          const err = r.headers.get("X-IVM-Preview-Error");
          if (err) {{
            const lbl = document.getElementById("lbl-" + img.getAttribute("data-idx"));
            if (lbl) lbl.textContent = cid + " (" + err + ")";
          }}
          return r.blob();
        }})
        .then(b => swapWhenReady(img, URL.createObjectURL(b)))
        .catch(() => {{}})
        .finally(() => {{
          clearTimeout(t);
          inflight--;
          drainQueue();
        }});
    }};
    if (inflight >= MAX_INFLIGHT) {{
      queue.push(run);
      return;
    }}
    run();
  }}

  CAMS.forEach((cid, idx) => {{
    const start = () => {{
      refreshOne(cid);
      setInterval(() => refreshOne(cid), PER_CAM_MS);
    }};
    setTimeout(start, idx * STAGGER_MS);
  }});
}})();
</script></body></html>"""
    return page, height_px
