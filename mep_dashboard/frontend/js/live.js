'use strict';

window.initLive = function () {
  // Linear display range: keep the project-scale baseline at 0..3 m and
  // expand only enough to include a larger real value.
  const Y_AXIS_DEFAULT_MIN = 0;
  const Y_AXIS_DEFAULT_MAX = 3;


  const chart = Dashboard.chart('position-chart', 'line', {
    datasets: [{
      label: 'Position Error (m)',
      data: [],
      borderColor: '#89e5fc',
      borderWidth: 2,
      pointRadius: 0,
      tension: .18,
      fill: false
    }]
  }, {
    scales: {
      x: Dashboard.axis('Simulation Time [s]', {
        type: 'linear'
      }),
      y: Dashboard.axis('Position Error [m]', {
        type: 'linear',
        min: Y_AXIS_DEFAULT_MIN,
        max: Y_AXIS_DEFAULT_MAX,
        ticks: {
          color: '#8d97a8',
          stepSize: 0.5
        }
      })
    }
  });

  const play = document.getElementById('play');
  const pause = document.getElementById('pause');
  const stop = document.getElementById('stop');

  const badge = document.getElementById('live-badge');
  const playback = document.getElementById('playback-status');

  const progressNumber = document.getElementById('progress');
  const progressTotalEl = document.getElementById('progress-total');
  const progressTrack = document.querySelector('.progress-track');

  const stateEl = document.getElementById('mission-state');
  const phaseEl = document.getElementById('mission-phase');

  const positionEl = document.getElementById('position-error');
  const orientationEl = document.getElementById('orientation-error');
  const velocityEl = document.getElementById('total-velocity');
  const angularEl = document.getElementById('total-angular-velocity');
  const remainingEl = document.getElementById('remaining-distance');
  const relativeVelocityEl = document.getElementById('relative-velocity');

  const signalEl = document.getElementById('live-signal');

  let socket = null;
  let reconnectTimer = null;

  function finite(value) {
    return typeof value === 'number' && Number.isFinite(value);
  }

  function fmt(value, digits = 3) {
    return finite(value) ? value.toFixed(digits) : '--';
  }

  function formatTime(seconds) {
    if (!finite(seconds)) return '00:00:00';

    const value = Math.max(0, Math.floor(seconds));

    return [
      Math.floor(value / 3600),
      Math.floor(value / 60) % 60,
      value % 60
    ]
      .map(part => String(part).padStart(2, '0'))
      .join(':');
  }

  function setProgress(value, total) {
    const totalSteps = Number(total) || progressTrack.querySelectorAll('i').length || 7;
    const progress = Math.max(0, Math.min(totalSteps, Number(value) || 0));

    progressNumber.textContent = progress;
    progressTotalEl.textContent = totalSteps;
    progressTrack.setAttribute('aria-valuemax', String(totalSteps));
    progressTrack.setAttribute('aria-valuenow', String(progress));

    [...progressTrack.querySelectorAll('i')].forEach((segment, index) => {
      segment.style.opacity = index < progress ? '1' : '.22';
    });

    document.querySelectorAll('.live-stage-list span').forEach((stage, index) => {
      stage.classList.toggle('active', index === progress - 1);
      stage.classList.toggle('complete', index < progress - 1);
    });
  }

  let missionControlMode = 'READY';
  let missionCommandBusy = false;
  let rosControlConnected = false;

  function updateMissionControls() {
    if (!rosControlConnected || missionCommandBusy) {
      play.disabled = true;
      pause.disabled = true;
      stop.disabled = true;
      return;
    }

    if (missionControlMode === 'READY') {
      play.textContent = '▶ PLAY';
      play.disabled = false;
      pause.disabled = true;
      stop.disabled = true;
      return;
    }

    if (missionControlMode === 'RUNNING') {
      play.textContent = '▶ PLAY';
      play.disabled = true;
      pause.disabled = false;
      stop.disabled = false;
      return;
    }

    if (missionControlMode === 'PAUSED') {
      play.textContent = '▶ RESUME';
      play.disabled = false;
      pause.disabled = true;
      stop.disabled = false;
      return;
    }

    // STOPPED / ABORTED
    play.textContent = '▶ PLAY';
    play.disabled = true;
    pause.disabled = true;
    stop.disabled = true;
  }

  function setConnection(connected) {
    rosControlConnected = Boolean(connected);
    if (connected) {
      badge.textContent = '● LIVE MISSION · ROS2 ONLINE';
      playback.textContent = 'ROS2 LIVE';
      signalEl.innerHTML = '<i class="dot"></i> LIVE ROS2 SIGNAL';

      updateMissionControls();
    } else {
      badge.textContent = '● LIVE MISSION · WAITING';
      playback.textContent = 'WAITING FOR ROS2';
      signalEl.innerHTML = '<i class="dot"></i> WAITING FOR SIGNAL';

      updateMissionControls();
    }

  }

  let lastGraphStage = null;

  function resetChartForStage(stage) {
    const data = chart.data.datasets[0].data;
    data.length = 0;
    delete chart.options.scales.x.min;
    delete chart.options.scales.x.max;
    chart.options.scales.y.min = Y_AXIS_DEFAULT_MIN;
    chart.options.scales.y.max = Y_AXIS_DEFAULT_MAX;
    chart.options.scales.y.ticks.stepSize = 0.5;
    lastGraphStage = stage;
  }

  function updateChart(simTime, error, missionStage) {
    if (!chart) return;

    const numericStage = Number(missionStage);
    const currentStage = Number.isInteger(numericStage) && numericStage > 0
      ? numericStage
      : null;
    let stageChanged = false;

    if (currentStage !== null && currentStage !== lastGraphStage) {
      resetChartForStage(currentStage);
      stageChanged = true;
    }

    if (!finite(simTime) || !finite(error) || error < 0) {
      if (stageChanged) chart.update('none');
      return;
    }

    const data = chart.data.datasets[0].data;

    data.push({
      x: simTime,
      y: error
    });

    while (data.length > 240) data.shift();

    if (data.length > 1) {
      chart.options.scales.x.min = data[0].x;
      chart.options.scales.x.max = data[data.length - 1].x;
    }

    const values = data.map(point => point.y).filter(y => finite(y) && y >= 0);
    if (values.length) {
      const actualMax = Math.max(...values);
      const axisMax = actualMax <= Y_AXIS_DEFAULT_MAX
        ? Y_AXIS_DEFAULT_MAX
        : Math.ceil(actualMax + 0.01);

      chart.options.scales.y.min = Y_AXIS_DEFAULT_MIN;
      chart.options.scales.y.max = axisMax;
      chart.options.scales.y.ticks.stepSize = axisMax <= 6 ? 0.5 : 1;
    }

    chart.update('none');
  }

  function render(data) {
    const connected = data.connected === true;

    setConnection(connected);
    setProgress(data.progress, data.total_steps);

    stateEl.textContent = data.phase || 'WAITING';
    stateEl.classList.toggle('state-failure', Boolean(data.is_failure));
    phaseEl.textContent = connected ? "ROS2 LIVE TELEMETRY" : "WAITING FOR ROS2";

    document.getElementById('mission-time').textContent =
      formatTime(data.sim_time_s);

    positionEl.textContent = fmt(data.position_error);
    orientationEl.textContent = fmt(data.orientation_error_deg, 2);

    velocityEl.textContent = fmt(data.total_velocity_mps);
    relativeVelocityEl.textContent = fmt(data.total_velocity_mps);

    angularEl.textContent = fmt(
      data.total_angular_velocity_deg_s,
      3
    );

    remainingEl.textContent = fmt(data.remaining_distance_m);

    updateChart(
      data.sim_time_s,
      data.position_error,
      data.progress
    );
  }

  /**
   * Generic "poll a JPEG endpoint into an <img>" live feed, shared by the
   * Isaac viewport and Camera 1/2/3 (they all work the same way: an <img>
   * that keeps re-requesting the same endpoint with a cache-busting query).
   */
  function initImageFeed({ headingIncludes, endpoint, elementId, alt, offlineAfterMs = 1500, refreshMs = 200, waitForImageLoad = false }) {
    const panels = [...document.querySelectorAll('article.panel')];
    const panel = panels.find(item => {
      const heading = item.querySelector('h2');
      return heading && heading.textContent.includes(headingIncludes);
    });

    if (!panel) {
      console.warn(`${headingIncludes} panel not found`);
      return;
    }

    const stream = panel.querySelector('.stream');
    const label = panel.querySelector('.stream-label');
    const badge = panel.querySelector('.offline');
    const meta = panel.querySelector('.stream-meta');

    if (!stream) {
      console.warn(`${headingIncludes} stream container not found`);
      return;
    }

    stream.style.position = 'relative';
    stream.style.overflow = 'hidden';

    let img = document.getElementById(elementId);

    if (!img) {
      img = document.createElement('img');
      img.id = elementId;
      img.alt = alt;

      img.style.position = 'absolute';
      img.style.inset = '0';
      img.style.width = '100%';
      img.style.height = '100%';
      img.style.objectFit = 'cover';
      img.style.zIndex = '0';

      stream.prepend(img);
    }

    function setStatus(value) {
      if (!meta) return;
      const labels = [...meta.querySelectorAll('dt')];
      const index = labels.findIndex(item => item.textContent.trim().toLowerCase() === 'status');
      const values = [...meta.querySelectorAll('dd')];
      if (index >= 0 && values[index]) values[index].textContent = value;
    }

    function setOnline() {
      if (label) label.style.display = 'none';

      if (badge) {
        badge.textContent = '● LIVE';
        badge.style.opacity = '1';
      }
      setStatus('LIVE');
    }

    function setOffline() {
      if (label) label.style.display = '';

      if (badge) {
        badge.textContent = '● WAITING';
      }
      setStatus('WAITING');
    }

    let objectUrl = null;

    async function refresh() {
      try {
        const response = await fetch(endpoint + '?t=' + Date.now(), { cache: 'no-store' });
        if (!response.ok || response.status === 204) throw new Error('NO FRAME');
        const nextUrl = URL.createObjectURL(await response.blob());
        img.src = nextUrl;
        if (waitForImageLoad && typeof img.decode === 'function') {
          try {
            await img.decode();
          } catch (error) {
            URL.revokeObjectURL(nextUrl);
            throw error;
          }
        }
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = nextUrl;
        setOnline();
        window.setTimeout(refresh, refreshMs);
      } catch (_) {
        setOffline();
        window.setTimeout(refresh, offlineAfterMs);
      }
    }

    setOffline();
    refresh();
  }

  function initViewport() {
    initImageFeed({
      headingIncludes: 'ISAAC SIM VIEWPORT',
      endpoint: '/api/live/viewport.jpg',
      elementId: 'isaac-viewport-image',
      alt: '',
      offlineAfterMs: 1500,
      refreshMs: 66,
      waitForImageLoad: true
    });
  }

  function initCamera1() {
    initImageFeed({
      headingIncludes: 'CAMERA 1',
      endpoint: '/api/live/camera1.jpg',
      elementId: 'camera1-live-image',
      alt: '',
      offlineAfterMs: 500
    });
  }

  function initCamera2() {
    initImageFeed({
      headingIncludes: 'CAMERA 2',
      endpoint: '/api/live/camera2.jpg',
      elementId: 'camera2-live-image',
      alt: '',
      offlineAfterMs: 500
    });
  }

  function initCamera3() {
    initImageFeed({
      headingIncludes: 'CAMERA 3',
      endpoint: '/api/live/camera3.jpg',
      elementId: 'camera3-live-image',
      alt: '',
      offlineAfterMs: 500
    });
  }

  function connect() {
    clearTimeout(reconnectTimer);

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';

    socket = new WebSocket(
      `${protocol}//${location.host}/ws/live`
    );

    socket.addEventListener('open', () => {
      playback.textContent = 'WEBSOCKET CONNECTED';
    });

    socket.addEventListener('message', event => {
      try {
        render(JSON.parse(event.data));
      } catch (error) {
        playback.textContent = 'INVALID LIVE DATA';
      }
    });

    socket.addEventListener('close', () => {
      setConnection(false);

      reconnectTimer = setTimeout(connect, 1500);
    });

    socket.addEventListener('error', () => {
      playback.textContent = 'WEBSOCKET ERROR';
    });
  }

  setProgress(0);
  setConnection(false);
  stateEl.textContent = 'WAITING FOR MISSION';
  phaseEl.textContent = 'ROS2 LIVE TELEMETRY';

  async function sendMissionCommand(command) {
    if (missionCommandBusy) {
      return false;
    }

    missionCommandBusy = true;
    updateMissionControls();

    playback.textContent =
      `SENDING ${command.toUpperCase()}...`;

    try {
      const response = await fetch(
        `/api/live/command/${command}`,
        {
          method: 'POST',
          headers: {
            'Accept': 'application/json'
          }
        }
      );

      let payload = {};

      try {
        payload = await response.json();
      } catch (_) {
        payload = {};
      }

      if (!response.ok) {
        throw new Error(
          payload.detail ||
          `HTTP ${response.status}`
        );
      }

      playback.textContent =
        `ROS2 ${command.toUpperCase()} SENT`;

      return true;

    } catch (error) {
      console.error(
        `Mission command '${command}' failed:`,
        error
      );

      playback.textContent =
        `COMMAND FAILED · ${command.toUpperCase()}`;

      return false;

    } finally {
      missionCommandBusy = false;
      updateMissionControls();
    }
  }


  play.addEventListener('click', async () => {
    if (missionControlMode === 'PAUSED') {
      const ok = await sendMissionCommand('resume');

      if (ok) {
        missionControlMode = 'RUNNING';
      }
    } else if (missionControlMode === 'READY') {
      const ok = await sendMissionCommand('start');

      if (ok) {
        missionControlMode = 'RUNNING';
      }
    }

    updateMissionControls();
  });


  pause.addEventListener('click', async () => {
    if (missionControlMode !== 'RUNNING') {
      return;
    }

    const ok = await sendMissionCommand('pause');

    if (ok) {
      missionControlMode = 'PAUSED';
    }

    updateMissionControls();
  });


  stop.addEventListener('click', async () => {
    if (
      missionControlMode !== 'RUNNING' &&
      missionControlMode !== 'PAUSED'
    ) {
      return;
    }

    const ok = await sendMissionCommand('abort');

    if (ok) {
      missionControlMode = 'STOPPED';
    }

    updateMissionControls();
  });


  updateMissionControls();


  initViewport();
  initCamera1();
  initCamera2();
  initCamera3();
  connect();
};
