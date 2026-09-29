'use strict';
// CAMERA 4: Astrobee satellite map. The accumulated voxel map (satellite frame) arrives as
// one binary frame per snapshot on the live WebSocket (/ws/live), forwarded by live.js as
// `astrobee-map-points`; the docking clearance as `astrobee-map-status`. Obstruction
// voxels (inside the docking corridor) are drawn red. The map grows as the Astrobee
// scans; an empty snapshot (new run) clears it. It turns slowly (redrawn ~10 times a
// second) until dragged / zoomed. Drag: orbit, wheel: zoom, double-click: re-frame.
// Until the first points arrive the panel stays a WAITING placeholder.
(function () {
  const MAGIC = 'ABM1';
  const FREE_COLOR = 0x9fb6cc;
  const OBSTRUCTION_COLOR = 0xff4d5e;

  const panel = [...document.querySelectorAll('article.panel')].find(item => {
    const heading = item.querySelector('h2');
    return heading && heading.textContent.includes('CAMERA 4');
  });
  if (!panel) return;
  const stream = panel.querySelector('.stream');
  const label = panel.querySelector('.stream-label');
  const badge = panel.querySelector('.offline');
  const meta = panel.querySelector('.stream-meta');

  function setMeta(name, value, tone) {
    if (!meta) return;
    const labels = [...meta.querySelectorAll('dt')];
    const index = labels.findIndex(item => item.textContent.trim().toLowerCase() === name);
    const cell = [...meta.querySelectorAll('dd')][index];
    if (!cell) return;
    cell.textContent = value;
    cell.classList.toggle('good', tone === 'good');
    cell.classList.toggle('bad', tone === 'bad');
    cell.classList.toggle('metric-unavailable', tone === 'warn');
  }

  // `live`: a map is drawn (else the placeholder); `alive`: the Astrobee is streaming
  // (backend liveness, false when the WebSocket drops). Badge like the other cameras.
  let live = null;
  let alive = false;
  function updateBadge() {
    if (badge) badge.textContent = live && alive ? '● LIVE' : '● WAITING';
  }
  function show(isLive) {
    if (isLive === live) return;
    live = isLive;
    if (label) label.style.display = isLive ? 'none' : '';
    if (canvas) canvas.hidden = !isLive;
    updateBadge();
  }

  function showClearance(status) {
    if (!status) return;
    if ('alive' in status) {
      alive = Boolean(status.alive);
      updateBadge();
      if (Object.keys(status).length === 1) return;  // liveness only
    }
    if (status.zone || status.reset) {
      lastZone = status.zone || null;
      zoneBlocked = status.is_clear === false && status.observed !== false;
      drawZone();
    } else if ('is_clear' in status) {
      zoneBlocked = status.is_clear === false && status.observed !== false;
      drawZone();
    }
    if (status.reset) setMeta('docking', '—');
    else if (!('is_clear' in status)) return;
    else if (status.observed === false) setMeta('docking', 'NOT OBSERVED', 'warn');
    else if (status.is_clear) setMeta('docking', 'DOCKING AVAILABLE', 'good');
    else setMeta('docking', 'DOCKING UNAVAILABLE', 'bad');
  }

  // three.js scene, created on the first snapshot (no WebGL cost while offline)
  let THREE = window.THREE;
  let renderer = null;
  let canvas = null;
  let scene, camera, freePoints, hitPoints;
  // Docking keep-out zone outline (from the clearance message): what counts as an obstacle
  let zoneLines = null;
  let lastZone = null;
  let zoneBlocked = false;
  const ZONE_CLEAR = 0x89e5fc;
  const ZONE_BLOCKED = 0xff4d5e;
  let target = null;
  let orbit = { azimuth: -2.3, elevation: 0.45, radius: 20 };
  let fitted = false;
  let userMoved = false;
  let needsRender = true;
  const SPIN_RAD_S = 0.09;  // turn rate [rad/s] (the earlier 0.0015 rad per frame at 60 fps)
  const SPIN_FPS = 10;
  let lastSpin = performance.now();

  function initScene() {
    if (renderer || !THREE) return !!renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      THREE = null;
      if (label) label.querySelector('span').textContent = 'WEBGL UNAVAILABLE';
      return false;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    canvas = renderer.domElement;
    canvas.hidden = true;
    Object.assign(canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', zIndex: '0', cursor: 'grab' });
    stream.style.position = 'relative';
    stream.style.overflow = 'hidden';
    stream.prepend(canvas);

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
    camera.up.set(0, 0, 1);
    freePoints = new THREE.Points(new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: FREE_COLOR, size: 0.045, sizeAttenuation: true }));
    hitPoints = new THREE.Points(new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: OBSTRUCTION_COLOR, size: 0.1, sizeAttenuation: true }));
    scene.add(freePoints, hitPoints);
    drawZone();
    target = new THREE.Vector3();

    new ResizeObserver(resize).observe(stream);
    bindControls();
    resize();
    requestAnimationFrame(loop);
    return true;
  }

  function resize() {
    if (!renderer) return;
    const width = Math.max(1, stream.clientWidth);
    const height = Math.max(1, stream.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    needsRender = true;
  }

  function placeCamera() {
    const { azimuth, elevation, radius } = orbit;
    camera.position.set(
      target.x + radius * Math.cos(elevation) * Math.cos(azimuth),
      target.y + radius * Math.cos(elevation) * Math.sin(azimuth),
      target.z + radius * Math.sin(elevation));
    camera.lookAt(target);
    needsRender = true;
  }

  function fit() {
    const geometry = freePoints.geometry.attributes.position ? freePoints.geometry : hitPoints.geometry;
    if (!geometry.attributes.position) return;
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere || !Number.isFinite(sphere.radius)) return;
    target.copy(sphere.center);
    orbit.radius = Math.max(1, sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.9);
    placeCamera();
  }

  function bindControls() {
    let drag = null;
    canvas.addEventListener('pointerdown', event => {
      drag = { x: event.clientX, y: event.clientY, azimuth: orbit.azimuth, elevation: orbit.elevation };
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = 'grabbing';
    });
    canvas.addEventListener('pointermove', event => {
      if (!drag) return;
      userMoved = true;
      orbit.azimuth = drag.azimuth - (event.clientX - drag.x) * 0.008;
      orbit.elevation = Math.max(-1.45, Math.min(1.45, drag.elevation + (event.clientY - drag.y) * 0.008));
      placeCamera();
    });
    const end = () => { drag = null; canvas.style.cursor = 'grab'; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      userMoved = true;
      orbit.radius = Math.max(0.5, Math.min(1500, orbit.radius * Math.exp(event.deltaY * 0.001)));
      placeCamera();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => { userMoved = false; fit(); });
  }

  function loop() {
    requestAnimationFrame(loop);
    // Slow turn until the user takes over, redrawn at most SPIN_FPS times a second (the
    // angle follows the clock, so the speed does not depend on it): redrawing the whole
    // cloud every animation frame kept the page busy and stalled the viewport stream
    if (!live || document.hidden) return;
    const now = performance.now();
    if (!userMoved && now - lastSpin >= 1000 / SPIN_FPS) {
      orbit.azimuth += SPIN_RAD_S * Math.min(now - lastSpin, 1000) / 1000;
      lastSpin = now;
      placeCamera();
    }
    if (needsRender) {
      renderer.render(scene, camera);
      needsRender = false;
    }
  }

  // Rings along the zone (nozzle interior profile + the cylinder in front of the exit)
  // and 8 lines along it, in the map (satellite) frame
  function drawZone() {
    if (!scene) return;
    if (zoneLines) {
      scene.remove(zoneLines);
      zoneLines.geometry.dispose();
      zoneLines = null;
    }
    const z = lastZone;
    if (!z || !Array.isArray(z.exit) || !Array.isArray(z.axis)) return;
    const exit = new THREE.Vector3(...z.exit);
    const axis = new THREE.Vector3(...z.axis).normalize();
    const ref = Math.abs(axis.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const e1 = new THREE.Vector3().crossVectors(axis, ref).normalize();
    const e2 = new THREE.Vector3().crossVectors(axis, e1);
    const rings = [];  // [depth along the axis (< 0: in front), radius]
    if (z.front_m > 0) rings.push([-z.front_m, z.front_radius_m], [0, z.front_radius_m]);
    (z.inner || []).forEach(([d, r]) => rings.push([d, r]));
    const at = (d, r, a) => exit.clone().addScaledVector(axis, d)
      .addScaledVector(e1, r * Math.cos(a)).addScaledVector(e2, r * Math.sin(a));
    const pts = [];
    const n = 32;
    rings.forEach(([d, r]) => {
      for (let i = 0; i < n; i++) pts.push(at(d, r, 2 * Math.PI * i / n), at(d, r, 2 * Math.PI * (i + 1) / n));
    });
    for (let k = 0; k < 8; k++) {
      const a = 2 * Math.PI * k / 8;
      for (let i = 0; i + 1 < rings.length; i++) pts.push(at(rings[i][0], rings[i][1], a), at(rings[i + 1][0], rings[i + 1][1], a));
    }
    zoneLines = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: zoneBlocked ? ZONE_BLOCKED : ZONE_CLEAR, transparent: true, opacity: 0.8 }));
    scene.add(zoneLines);
    needsRender = true;
  }

  function setPoints(points, positions) {
    const geometry = new THREE.BufferGeometry();
    if (positions.length) geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    points.geometry.dispose();
    points.geometry = geometry;
  }

  function onFrame(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 12) return;
    const view = new DataView(buffer);
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== MAGIC) return;
    const count = view.getUint32(8, true);
    if (buffer.byteLength < 12 + count * 13) return;
    if (count === 0) {  // new run: start from an empty map
      if (renderer) {
        setPoints(freePoints, new Float32Array(0));
        setPoints(hitPoints, new Float32Array(0));
      }
      fitted = false;
      userMoved = false;
      show(false);
      return;
    }
    if (!initScene()) return;
    const positions = new Float32Array(buffer, 12, count * 3);
    const flags = new Uint8Array(buffer, 12 + count * 12, count);
    let hits = 0;
    for (let i = 0; i < count; i++) hits += flags[i] ? 1 : 0;
    const free = new Float32Array((count - hits) * 3);
    const hit = new Float32Array(hits * 3);
    for (let i = 0, f = 0, h = 0; i < count; i++) {
      const out = flags[i] ? hit : free;
      const k = flags[i] ? h++ : f++;
      out[3 * k] = positions[3 * i];
      out[3 * k + 1] = positions[3 * i + 1];
      out[3 * k + 2] = positions[3 * i + 2];
    }
    setPoints(freePoints, free);
    setPoints(hitPoints, hit);
    show(true);
    if (!fitted || !userMoved) { fit(); fitted = true; }  // follow the growing map until the user takes over
    needsRender = true;
  }

  window.addEventListener('astrobee-map-points', event => onFrame(event.detail));
  window.addEventListener('astrobee-map-status', event => showClearance(event.detail));
  show(false);
})();
