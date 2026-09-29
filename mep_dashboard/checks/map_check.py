"""CAMERA 4 · Astrobee map panel: smoke check without ROS 2 and without map data.

1. Server side (FastAPI TestClient, no network): the page carries the 4th camera panel in
   the same markup as the other three, three.js is served locally, the live WebSocket
   without ROS sends only the offline JSON (no map field, no binary frame), a stubbed
   bridge delivers the `astrobee_map` field + one binary snapshot frame, and the run
   point cloud endpoint serves `<id>.ply` like the run video.
2. Browser (only if Playwright has a Chromium): the panel renders as the OFFLINE
   placeholder with no WS map data, and stacks within a phone width.

    .venv/bin/python checks/map_check.py
"""
import json as _json_mod
import os
import struct
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

VIDEO_DIR = tempfile.mkdtemp(prefix="mapcheck-")
os.environ["MRV_RUN_VIDEO_DIR"] = VIDEO_DIR
os.environ.setdefault("MEP_DASHBOARD_CACHE_DIR", tempfile.mkdtemp(prefix="mapcheck-cache-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as A  # noqa: E402

ok = True


def check(label, cond, extra=""):
    global ok
    ok = ok and bool(cond)
    print(("PASS " if cond else "FAIL ") + label + ((" -> " + str(extra)) if extra else ""))


class Panels(HTMLParser):
    """Class lists and <h2> text of each `article.camera` (camera row), and the scripts."""

    def __init__(self):
        super().__init__()
        self.stack, self.panels, self.row_class, self.scripts = [], [], None, []

    def _in_camera(self):
        return any(cam for _, cam in self.stack)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "script" and a.get("src"):
            self.scripts.append(a["src"].split("?")[0])
        if "camera-row" in cls:
            self.row_class = cls
        cam = tag == "article" and "camera" in cls
        if cam:
            self.panels.append({"classes": [], "h2": ""})
        if cam or self._in_camera():
            self.panels[-1]["classes"] += cls
        if tag not in ("meta", "link", "br", "img", "input"):
            self.stack.append((tag, cam))

    def handle_endtag(self, tag):
        while self.stack:
            t, _ = self.stack.pop()
            if t == tag:
                break

    def handle_data(self, data):
        if self.stack and self.stack[-1][0] == "h2" and self._in_camera():
            self.panels[-1]["h2"] += data


## 1. Server side
A.app.router.on_startup.clear()  # never touch ROS 2 in this check
A.live_ros, A.live_ros_error = None, "ROS2 disabled for map_check"
client = TestClient(A.app)

html = client.get("/").text
parser = Panels()
parser.feed(html)
titles = [p["h2"].strip() for p in parser.panels]
check("camera row has 4 panels, map next to CAMERA 3",
      titles == ["CAMERA 1 · MEP CAPTURE", "CAMERA 2 · SATELLITE DOCKING", "CAMERA 3 · ASTROBEE", "CAMERA 4 · ASTROBEE MAP"], titles)
check("camera row uses the 4-column layout", parser.row_class and "camera-row-4" in parser.row_class, parser.row_class)
same = {"panel-heading", "offline", "stream", "reticle", "stream-label", "stream-meta"}
check("map panel has the same markup as the camera panels",
      all(same <= set(p["classes"]) for p in parser.panels), [sorted(same - set(p["classes"])) for p in parser.panels])
scripts = parser.scripts
check("three.js + map3d.js load locally, before live.js",
      "/static/js/vendor/three.min.js" in scripts and "/static/js/map3d.js" in scripts
      and scripts.index("/static/js/map3d.js") < scripts.index("/static/js/live.js")
      and all(not s.startswith("http") for s in scripts), scripts)
three = client.get("/static/js/vendor/three.min.js")
check("vendored three.js is served", three.status_code == 200 and b"THREE" in three.content[:2000], f"{len(three.content)} bytes")
css = client.get("/static/css/style.css").text
check("index is revalidated, assets are versioned",
      client.get("/").headers.get("cache-control") == "no-cache" and "style.css?v=" in html)
check("css has the 4-column row and its phone-width stacking",
      ".camera-row-4{grid-template-columns:repeat(4" in css and ".camera-row-4{grid-template-rows:repeat(4,280px)}" in css)

with client.websocket_connect("/ws/live") as ws:
    first = ws.receive_json()
check("ws without ROS: offline JSON, no map field", first.get("connected") is False and "astrobee_map" not in first, first)


class StubBridge:
    """The three calls the WS loop makes, with one map snapshot + clearance."""

    def __init__(self):
        xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
        self.frame = A.encode_map_frame(1, xyz, [False, True, False])
        self.status = {"is_clear": False, "observed": True, "status": "DOCKING_UNAVAILABLE", "obstruction_voxels": 1,
                       "nearest_m": 0.3, "points": 3, "map_seq": 1,
                       "zone": {"exit": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "front_m": 1.0, "front_radius_m": 0.6,
                                "inner": [[0.0, 0.35], [0.3, 0.33], [0.6, 0.3]]}}

    def snapshot(self):
        return 1, {**A.normalize_live_status({"state": "DOCKING_UNAVAILABLE", "sim_time_s": 42.0}, last_progress=2),
                   "astrobee_map": self.status}

    def map_status(self):
        return 1, self.status

    def map_frame(self):
        return 1, self.frame

    alive = True

    def astrobee_alive(self):
        return self.alive

    # the page also polls the camera feeds: none in this check
    camera1_snapshot = camera2_snapshot = camera3_snapshot = viewport_snapshot = lambda self: None


A.live_ros = StubBridge()
telemetry = status = frame = alive_msg = None
with client.websocket_connect("/ws/live") as ws:
    for _ in range(4):  # telemetry, clearance, liveness (JSON) and the snapshot (binary), in any order
        msg = ws.receive()
        if msg.get("bytes") is not None:
            frame = msg["bytes"]
            continue
        data = _json_mod.loads(msg["text"])
        if data.get("type") != "astrobee_map":
            telemetry = data
        elif set(data["astrobee_map"]) == {"alive"}:
            alive_msg = data["astrobee_map"]
        else:
            status = data
A.live_ros = None
check("liveness message (CAMERA 4 LIVE / WAITING)", alive_msg == {"alive": True}, alive_msg)
check("telemetry payload carries the astrobee_map field", telemetry.get("astrobee_map", {}).get("status") == "DOCKING_UNAVAILABLE")
check("map status message", status.get("type") == "astrobee_map" and status["astrobee_map"]["obstruction_voxels"] == 1)
magic, (seq, n) = frame[:4], struct.unpack("<II", frame[4:12])
xyz = np.frombuffer(frame, dtype="<f4", count=3 * n, offset=12).reshape(n, 3)
flags = np.frombuffer(frame, dtype=np.uint8, count=n, offset=12 + 12 * n)
check("binary snapshot frame: ABM1, seq, float32 xyz, uint8 flags",
      magic == b"ABM1" and (seq, n) == (1, 3) and np.allclose(xyz[1], [1, 2, 3]) and flags.tolist() == [0, 1, 0]
      and len(frame) == 12 + 13 * n)
big = np.random.default_rng(0).normal(size=(5000, 3))
hits = np.zeros(5000, dtype=bool)
hits[::997] = True
thin = A.encode_map_frame(2, big, hits, max_points=1000)
n_thin = struct.unpack("<I", thin[8:12])[0]
kept_flags = np.frombuffer(thin, dtype=np.uint8, count=n_thin, offset=12 + 12 * n_thin)
check("large maps are thinned, obstruction voxels kept", n_thin == 1000 and int(kept_flags.sum()) == int(hits.sum()), n_thin)

# a new run (the sim clock goes back) clears the previous run's map
import json as _json
import threading as _threading
bridge = A.RosLiveBridge.__new__(A.RosLiveBridge)
bridge._lock, bridge._latest, bridge._seq = _threading.Lock(), None, 0
bridge._map_frame, bridge._map_seq, bridge._map_points = None, 0, 0
bridge._map_status, bridge._map_status_seq, bridge._last_sim_time = None, 0, None
status = lambda t: type("M", (), {"data": _json.dumps({"state": "SEARCH", "sim_time_s": t})})()  # noqa: E731
bridge._status_callback(status(150.0))
bridge._map_frame, bridge._map_seq, bridge._map_points = A.encode_map_frame(1, np.ones((5, 3)), np.zeros(5, bool)), 1, 5
bridge._map_status = {"is_clear": True}
bridge._status_callback(status(151.0))
kept = bridge.map_frame()[1]
bridge._status_callback(status(0.5))
seq_after, cleared = bridge.map_frame()
check("new run clears the map view (empty frame, no verdict)",
      struct.unpack("<I", kept[8:12])[0] == 5 and struct.unpack("<I", cleared[8:12])[0] == 0 and seq_after == 2
      and "is_clear" not in bridge.map_status()[1])

missing = client.get("/api/validation/runs/run_nomap/pointcloud")
check("point cloud 404 when the run has none", missing.status_code == 404)
check("point cloud rejects bad ids", client.get("/api/validation/runs/..%2Fx/pointcloud").status_code in (400, 404))
Path(VIDEO_DIR, "run_map.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
got = client.get("/api/validation/runs/run_map/pointcloud")
head = client.head("/api/validation/runs/run_map/pointcloud")
check("point cloud served like the run video", got.status_code == 200 and got.content.startswith(b"ply\n") and head.status_code == 200,
      got.headers.get("content-disposition"))

## 2. Browser (optional)
url = None
try:
    from playwright.sync_api import sync_playwright

    import threading

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(A.app, host="127.0.0.1", port=int(os.environ.get("MAP_CHECK_PORT", "8765")), log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # no browser installed: `playwright install chromium`
            print(f"SKIP browser part: {str(exc).splitlines()[0]}")
            browser = None
        if browser is not None:
            thread.start()
            while not server.started:
                pass
            url = f"http://127.0.0.1:{server.config.port}"
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(1500)
            map_panel = page.locator("article.panel", has_text="CAMERA 4")
            check("browser: map panel present", map_panel.count() == 1)
            check("browser: WAITING badge + placeholder label without data",
                  map_panel.locator(".offline").inner_text().strip() == "● WAITING"
                  and map_panel.locator(".stream-label").is_visible(), map_panel.locator(".stream-meta").inner_text())
            check("browser: no map canvas before data", map_panel.locator("canvas:visible").count() == 0)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(300)
            box = map_panel.bounding_box()
            # (the page header itself is wider than 390 px at HEAD; only the map panel is judged here)
            check("browser: map panel stacks within 390 px", box and box["x"] >= 0 and box["x"] + box["width"] <= 390, box)
            check("browser: no page / console errors", not errors, errors)
            # ... and with a (stubbed) map on the WebSocket: canvas, LIVE, verdict in the stream meta
            A.live_ros = StubBridge()
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.goto(url, wait_until="domcontentloaded")
            map_panel = page.locator("article.panel", has_text="CAMERA 4")
            try:
                page.wait_for_function("[...document.querySelectorAll('article.panel')].some(p => "
                                       "p.textContent.includes('CAMERA 4') && p.querySelector('canvas:not([hidden])'))", timeout=15_000)
            except Exception:
                pass
            meta = map_panel.locator(".stream-meta").inner_text()
            check("browser: map snapshot drawn (LIVE + canvas)",
                  map_panel.locator("canvas:visible").count() == 1 and "LIVE" in map_panel.locator(".offline").inner_text(), meta)
            check("browser: stream meta shows the docking verdict only",
                  meta.split() == ["Docking", "DOCKING", "UNAVAILABLE"]
                  and not map_panel.locator(".stream-label").is_visible(), meta)
            # the Astrobee stops streaming -> WAITING (the last map stays drawn)
            A.live_ros.alive = False
            try:
                page.wait_for_function("[...document.querySelectorAll('article.panel')].find(p => p.textContent.includes('CAMERA 4'))"
                                       ".querySelector('.offline').textContent === '● WAITING'", timeout=5_000)
            except Exception:
                pass
            check("browser: link lost -> CAMERA 4 WAITING, map kept",
                  map_panel.locator(".offline").inner_text().strip() == "● WAITING" and map_panel.locator("canvas:visible").count() == 1)
            A.live_ros.alive = True
            state = page.locator("#mission-state").inner_text()
            phase = page.locator("#mission-phase").inner_text()
            check("browser: mission panel shows the stop (DOCKING UNAVAILABLE, stage kept, red)",
                  state == "DOCKING UNAVAILABLE" and phase == "ASTROBEE: DOCKING PORT BLOCKED"
                  and "state-failure" in (page.locator("#mission-state").get_attribute("class") or ""), f"{state} | {phase}")
            page.screenshot(path=str(Path(__file__).resolve().parent / "map-live.png"))
            A.live_ros = None
            browser.close()
            server.should_exit = True
except ImportError as exc:
    print(f"SKIP browser part: {exc}")

print("\nRESULT:", "ALL PASS" if ok else "FAILURES", "" if url else "(server side only)")
sys.exit(0 if ok else 1)
