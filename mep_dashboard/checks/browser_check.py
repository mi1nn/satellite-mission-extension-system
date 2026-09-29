"""Browser smoke check for the live ROS2 + Firestore dashboard."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
URL = os.environ.get("DASHBOARD_URL", "http://127.0.0.1:8000")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page_errors = []
    console_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)

    response = page.goto(URL, wait_until="domcontentloaded")
    assert response and response.status == 200
    assert page.locator("#reset").count() == 0
    assert page.locator(".progress-track i").count() == 7
    assert page.locator(".live-stage-list span").count() == 7
    assert page.locator("#play").count() == 1
    assert page.locator("#pause").count() == 1
    assert page.locator("#stop").count() == 1
    # CAMERA 4 (Astrobee map) next to CAMERA 3; OFFLINE placeholder until map data arrives
    cameras = page.locator(".camera-row-4 article.camera h2").all_inner_texts()
    assert cameras[2:] == ["CAMERA 3 · ASTROBEE", "CAMERA 4 · ASTROBEE MAP"], cameras
    map_panel = page.locator("article.panel", has_text="CAMERA 4")
    if map_panel.locator("canvas:visible").count() == 0:
        assert map_panel.locator(".offline").inner_text().strip() == "● WAITING"
        assert map_panel.locator(".stream-label").is_visible()

    page.locator("#validation-tab").click()
    page.wait_for_function(
        "!document.querySelector('#validation-status').textContent.startsWith('Loading')",
        timeout=30_000,
    )
    assert "FIREBASE CONNECTION ERROR" not in page.locator("#validation-status").inner_text()
    run_count = page.locator("#run-history tr").count()
    assert run_count > 0
    page.wait_for_function("document.querySelectorAll('#stage-selector button').length === 7", timeout=30_000)
    assert page.locator(".kpi-row .kpi").count() == 5
    assert page.locator("#stage-selector button").count() == 7
    assert page.locator('[data-stage="7"]').count() == 1
    assert "Mission" not in page.locator(".history thead").inner_text()
    assert page.locator("#selected-run-id").inner_text().startswith("Run ID:")

    # The selected Run may be incomplete, but any available metric must produce
    # a real chart/range/export state without NaN or undefined text.
    page.wait_for_timeout(1_000)
    metric_count = page.locator("#metric-selector button").count()
    if metric_count:
        assert page.locator("#validation-telemetry-chart").is_visible()
        assert page.locator("#range-controls").is_visible()
        assert page.locator("#export-json").is_enabled()

    panel_text = page.locator("#validation-panel").inner_text()
    assert "NaN" not in panel_text
    assert "undefined" not in panel_text
    page.screenshot(path=str(OUT / "validation-current.png"), full_page=True)

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") == 390
    assert not page_errors, page_errors
    assert not console_errors, console_errors

    report = {
        "result": "PASS",
        "url": URL,
        "run_count": run_count,
        "stage_count": 7,
        "metric_count": metric_count,
        "page_errors": page_errors,
        "console_errors": console_errors,
    }
    (OUT / "report-current.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    browser.close()
