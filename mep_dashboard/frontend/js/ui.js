/*
 * Decorative / convenience UI (no data logic):
 *  - dot globe behind the Isaac viewport while its stream is offline
 *  - round "↗" button in each card header: show that card fullscreen (again / Esc: back)
 */
(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---------------------------------------------------------------- dot globe
  function initGlobe() {
    const label = document.querySelector('.viewport .stream-label');
    if (!label) return;
    const holder = document.createElement('div');
    holder.className = 'globe';
    holder.setAttribute('aria-hidden', 'true');
    const canvas = document.createElement('canvas');
    holder.appendChild(canvas);
    label.prepend(holder);  // hidden together with the label once the stream is live
    const ctx = canvas.getContext('2d');

    // Fibonacci sphere; a cheap value-noise "land" mask gives continents-like clusters
    const N = 2600;
    const points = [];
    const hash = (x, y, z) => {
      const s = Math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453;
      return s - Math.floor(s);
    };
    const land = (x, y, z) => {
      const f = 2.2;
      let v = 0;
      for (let o = 0; o < 3; o += 1) {
        const k = f * (o + 1) * 1.7;
        v += (Math.sin(x * k + 1.3 * o) * Math.cos(y * k * 1.1 - o) + Math.sin(z * k * 0.9 + 2.1 * o)) / (o + 1);
      }
      return v;
    };
    for (let i = 0; i < N; i += 1) {
      const y = 1 - (i / (N - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const th = i * Math.PI * (3 - Math.sqrt(5));
      const x = Math.cos(th) * r;
      const z = Math.sin(th) * r;
      const isLand = land(x, y, z) > 0.35;
      points.push({ x, y, z, land: isLand, tw: hash(x, y, z) });
    }
    // A few "satellite" markers on the surface
    const markers = [[0.35, 0.45], [-0.5, 0.2], [0.1, -0.35], [0.75, -0.1], [-0.2, 0.7]].map(([lon, lat]) => ({
      x: Math.cos(lat) * Math.cos(lon * Math.PI), y: Math.sin(lat), z: Math.cos(lat) * Math.sin(lon * Math.PI),
    }));

    let w = 0; let h = 0; let dpr = 1;
    const resize = () => {
      const box = holder.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = Math.max(1, box.width); h = Math.max(1, box.height);
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    };
    new ResizeObserver(resize).observe(holder);
    resize();

    const draw = (t) => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const R = Math.min(w, h) * 0.46;
      const cx = w * 0.5; const cy = h * 0.58;
      const rot = reduceMotion ? 0.6 : t * 0.00006;
      const tilt = 0.32;
      const cr = Math.cos(rot); const sr = Math.sin(rot);
      const ct = Math.cos(tilt); const st = Math.sin(tilt);
      const project = (p) => {
        const x1 = p.x * cr + p.z * sr;
        const z1 = -p.x * sr + p.z * cr;
        const y1 = p.y * ct - z1 * st;
        const z2 = p.y * st + z1 * ct;
        return { x: cx + x1 * R, y: cy - y1 * R, z: z2 };
      };

      // Atmosphere glow
      const g = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 1.25);
      g.addColorStop(0, 'rgba(104,137,247,0.10)');
      g.addColorStop(0.75, 'rgba(137,229,252,0.10)');
      g.addColorStop(1, 'rgba(137,229,252,0)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(cx, cy, R * 1.25, 0, Math.PI * 2); ctx.fill();
      // Limb
      ctx.strokeStyle = 'rgba(137,229,252,0.22)';
      ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();

      // Dots (front hemisphere only)
      for (const p of points) {
        const q = project(p);
        if (q.z < 0) continue;
        const light = 0.25 + 0.75 * q.z;
        const size = p.land ? 1.35 : 0.8;
        const alpha = p.land ? 0.35 + 0.6 * light * (0.7 + 0.3 * Math.sin(t * 0.002 + p.tw * 6.28)) : 0.12 * light;
        ctx.fillStyle = p.land ? `rgba(160,225,252,${alpha.toFixed(3)})` : `rgba(104,137,247,${alpha.toFixed(3)})`;
        ctx.fillRect(q.x - size / 2, q.y - size / 2, size, size);
      }

      // Orbit ellipse
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-0.28);
      ctx.strokeStyle = 'rgba(137,229,252,0.18)';
      ctx.setLineDash([2, 6]);
      ctx.beginPath(); ctx.ellipse(0, 0, R * 1.35, R * 0.42, 0, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      const a = reduceMotion ? 1 : t * 0.00035;
      const sx = Math.cos(a) * R * 1.35; const sy = Math.sin(a) * R * 0.42;
      ctx.fillStyle = 'rgba(137,229,252,0.95)';
      ctx.shadowColor = 'rgba(137,229,252,0.9)'; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(sx, sy, 3.2, 0, Math.PI * 2); ctx.fill();
      ctx.restore();

      // Pulsing markers
      markers.forEach((m, i) => {
        const q = project(m);
        if (q.z < 0.1) return;
        const pulse = reduceMotion ? 0.5 : (t * 0.0006 + i * 0.37) % 1;
        ctx.strokeStyle = `rgba(137,229,252,${(0.5 * (1 - pulse)).toFixed(3)})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(q.x, q.y, 5 + pulse * 18, 0, Math.PI * 2); ctx.stroke();
        ctx.strokeStyle = 'rgba(137,229,252,0.35)';
        ctx.beginPath(); ctx.arc(q.x, q.y, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#89e5fc';
        ctx.shadowColor = 'rgba(137,229,252,0.9)'; ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(q.x, q.y, 3, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      });
    };

    const frame = (t) => {
      // Only render while visible (the label hides when the live stream is shown)
      if (holder.offsetParent !== null && !document.hidden) draw(t);
      window.setTimeout(() => window.requestAnimationFrame(frame), reduceMotion ? 1000 : 33);
    };
    window.requestAnimationFrame(frame);
  }

  // ------------------------------------------------------- card expand buttons
  function initExpandButtons() {
    const cards = document.querySelectorAll('.panel > .panel-heading');
    cards.forEach(heading => {
      const card = heading.closest('.panel');
      if (!card || heading.querySelector('.icon-btn')) return;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'icon-btn';
      button.textContent = '↗';
      const title = heading.querySelector('h2')?.textContent.trim() || 'panel';
      button.setAttribute('aria-label', `Expand ${title}`);
      button.title = 'Expand';
      button.addEventListener('click', async () => {
        try {
          if (document.fullscreenElement === card) await document.exitFullscreen();
          else await card.requestFullscreen();
        } catch (_) { /* fullscreen not allowed */ }
      });
      heading.appendChild(button);
    });
    document.addEventListener('fullscreenchange', () => {
      document.querySelectorAll('.icon-btn').forEach(button => {
        const expanded = document.fullscreenElement === button.closest('.panel');
        button.textContent = expanded ? '↙' : '↗';
        button.title = expanded ? 'Close' : 'Expand';
        button.setAttribute('aria-pressed', String(expanded));
      });
    });
  }

  const start = () => { initGlobe(); initExpandButtons(); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
