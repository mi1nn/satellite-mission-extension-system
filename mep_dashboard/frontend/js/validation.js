'use strict';

window.initValidation = async function () {
  const $ = id => document.getElementById(id);
  const hasBool = value => typeof value === 'boolean';
  const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
  const text = value => value === null || value === undefined || value === '' ? '—' : String(value);
  const esc = value => text(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[char]);
  const number = (value, unit = '', digits = 3) => finite(value) === null ? '—' : `${value.toFixed(digits)}${unit}`;
  const date = value => {
    if (!value) return '—';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? '—' : parsed.toISOString().slice(0, 19).replace('T', ' ');
  };
  const bool = value => hasBool(value) ? value : null;
  const timeOf = row => finite(row.sim_time) ?? finite(row.sim_time_s);


  const STAGES = [
    {
      index: 1,
      key: 'MRV_ASTROBEE_MOVE',
      label: 'MRV 이동 + Astrobee 이동',
      states: ['INIT', 'MRV_MOVE_STEP_1', 'MRV_STEP_1_REACHED', 'MRV_MOVE_STEP_2', 'MRV_STEP_2_REACHED', 'ARM_DEPLOY'],
      metrics: [
        ['ee_x', 'End-effector X', ' m'],
        ['ee_y', 'End-effector Y', ' m'],
        ['ee_z', 'End-effector Z', ' m']
      ]
    },
    {
      index: 2,
      key: 'MEP_SEARCH',
      label: 'MEP 탐색',
      states: ['SEARCH', 'TAG_DETECTED', 'POSE_ESTIMATED', 'PREDICTING', 'APPROACHING', 'SLOW_APPROACH', 'CAPTURE_ATTEMPT', 'TAG_LOST', 'POSE_INVALID', 'PREDICTION_INVALID', 'APPROACH_TIMEOUT', 'CAPTURE_FAILED', 'PHYSICS_ERROR'],
      metrics: [
        ['distance_m', 'Capture Distance', ' m'],
        ['lateral_error_mm', 'Lateral Error', ' mm'],
        ['angle_error_deg', 'Orientation Error', ' deg'],
        ['rel_vel_mps', 'Relative Velocity', ' m/s'],
        ['rel_ang_vel_rad_s', 'Relative Angular Velocity', ' rad/s']
      ]
    },
    {
      index: 3,
      key: 'MEP_ATTACH',
      label: 'MEP 부착',
      states: ['CAPTURED', 'HOLDING', 'RETREAT'],
      metrics: [
        ['distance_m', 'Capture Distance', ' m'],
        ['lateral_error_mm', 'Lateral Error', ' mm'],
        ['angle_error_deg', 'Orientation Error', ' deg'],
        ['rel_vel_mps', 'Relative Velocity', ' m/s'],
        ['rel_ang_vel_rad_s', 'Relative Angular Velocity', ' rad/s']
      ]
    },
    {
      index: 4,
      key: 'DOCK_PREP',
      label: '도킹 준비',
      states: ['DOCK_TARGET_ACQUIRE', 'PRE_DOCK_APPROACH', 'XY_ALIGN', 'ORIENTATION_ALIGN', 'ALIGNMENT_CHECK', 'CLIENT_RELEASE', 'CLIENT_CRUISE', 'CHASE', 'VELOCITY_MATCHING', 'RENDEZVOUS', 'CLIENT_RELEASE_FAILED', 'VELOCITY_MATCH_TIMEOUT', 'RENDEZVOUS_TIMEOUT'],
      metrics: [
        ['dock_geometry_distance_m', 'Docking Distance', ' m'],
        ['dock_lateral_error_m', 'Lateral Error', ' m'],
        ['dock_orientation_error_deg', 'Orientation Error', ' deg'],
        ['dock_relative_speed_mps', 'Relative Velocity', ' m/s'],
        ['dock_insertion_depth_m', 'Insertion Depth', ' m']
      ]
    },
    {
      index: 5,
      key: 'DOCKING',
      label: '도킹',
      states: ['Z_APPROACH', 'FINAL_INSERTION', 'DOCK_READY', 'DOCKED', 'DOCK_FAILED'],
      metrics: [
        ['dock_geometry_distance_m', 'Docking Distance', ' m'],
        ['dock_lateral_error_m', 'Lateral Error', ' m'],
        ['dock_orientation_error_deg', 'Orientation Error', ' deg'],
        ['dock_relative_speed_mps', 'Relative Velocity', ' m/s'],
        ['dock_insertion_depth_m', 'Insertion Depth', ' m']
      ]
    },
    {
      index: 6,
      key: 'DOCK_COMPLETE',
      label: '도킹 완료',
      states: ['DOCK_HOLDING', 'STABILIZING', 'STOPPING', 'ROBOT_RELEASE', 'ARM_RETREAT', 'MRV_SEPARATION', 'STOP_FAILED', 'ROBOT_RELEASE_FAILED', 'ARM_RETREAT_FAILED', 'SEPARATION_COLLISION', 'SEPARATION_FAILED'],
      metrics: [
        ['dock_geometry_distance_m', 'Final Docking Error', ' m'],
        ['dock_lateral_error_m', 'Lateral Error', ' m'],
        ['dock_orientation_error_deg', 'Orientation Error', ' deg'],
        ['dock_relative_speed_mps', 'Relative Velocity', ' m/s'],
        ['dock_insertion_depth_m', 'Insertion Depth', ' m']
      ]
    },
    {
      index: 7,
      key: 'ORBIT_TRANSFER',
      label: '추진체 궤도 이송',
      states: [],
      metrics: []
    }
  ];

  const stateToStage = new Map();
  STAGES.forEach(stage => stage.states.forEach(state => stateToStage.set(state, stage.index)));

  const normalizeRun = source => {
    const captureMetricVersion = typeof source.capture_metric_version === 'string'
      ? source.capture_metric_version
      : null;
    const hasGtCaptureMetric = captureMetricVersion === 'gt_target_v1' &&
      finite(source.capture_gt_position_error_m) !== null;

    return {
      raw: source,
      sessionId: typeof source.session_id === 'string'
        ? source.session_id
        : typeof source.id === 'string' ? source.id : null,
      createdAt: typeof source.created_at === 'string' ? source.created_at : null,
      duration: finite(source.duration_sec),
      captureSuccess: bool(source.capture_success),
      dockingSuccess: bool(source.docking_success),
      missionSuccess: bool(source.mission_success),
      captureMetricVersion,
      hasGtCaptureMetric,
      captureError: hasGtCaptureMetric ? finite(source.capture_gt_position_error_m) : null,
      captureLateralError: hasGtCaptureMetric ? finite(source.capture_gt_lateral_error_m) : null,
      captureOrientationError: hasGtCaptureMetric ? finite(source.capture_gt_orientation_error_deg) : null,
      captureRelativeVelocity: hasGtCaptureMetric ? finite(source.capture_gt_relative_velocity_mps) : null,
      captureRelativeAngularVelocity: hasGtCaptureMetric ? finite(source.capture_gt_relative_angular_velocity_rad_s) : null,
      captureGap: hasGtCaptureMetric ? finite(source.capture_gt_gap_m) : null,
      captureRemainingDistance: hasGtCaptureMetric ? finite(source.capture_gt_remaining_distance_m) : null,
      captureTargetGap: hasGtCaptureMetric ? finite(source.capture_target_gap_m) : null,
      legacyCapturePositionError: finite(source.capture_position_error_m),
      dockingError: finite(source.docking_position_error_m),
      dockingLateralError: finite(source.docking_lateral_error_m),
      dockingOrientationError: finite(source.docking_orientation_error_deg),
      dockingInsertionDepth: finite(source.docking_insertion_depth_m),
      dockingRelativeVelocity: finite(source.docking_relative_velocity_mps),
      finalState: source.final_state ?? null,
      failureStage: source.failure_stage ?? null,
      failureReason: source.failure_reason ?? null,
      isRunning: bool(source.is_running)
    };
  };

  // Whole pipeline passed: reached the SUCCESS terminal state with both capture and
  // docking done. `mission_success` alone is not enough -- it is true for capture-only
  // runs (no --dock) and for sessions closed by the idle timeout after docking.
  const fullPipelineSuccess = run =>
    run.raw.final_state === 'SUCCESS' && run.captureSuccess === true && run.dockingSuccess === true;
  const values = (runs, key) => runs.map(run => run[key]).filter(value => finite(value) !== null);
  const mean = list => list.length ? list.reduce((sum, value) => sum + value, 0) / list.length : null;
  const successRate = list => {
    const known = list.filter(hasBool);
    const count = known.filter(Boolean).length;
    return known.length ? { rate: count / known.length, count, total: known.length } : null;
  };
  const percent = result => result ? `${(result.rate * 100).toFixed(1)}%` : 'N/A';
  // Repeatability: spread (population standard deviation) of the error over repeated runs.
  const spread = errors => {
    if (!errors.length) return null;
    const avg = mean(errors);
    const sigma = Math.sqrt(mean(errors.map(value => (value - avg) ** 2)));
    return { mean: avg, sigma, total: errors.length };
  };
  const formatSpread = result => result ? `±${(result.sigma * 1000).toFixed(2)} mm` : 'N/A';
  const spreadNote = result => `${result.total} successful run${result.total === 1 ? '' : 's'}`;
  const formatDuration = seconds => {
    if (finite(seconds) === null) return 'N/A';
    const rounded = Math.max(0, Math.round(seconds));
    return `${Math.floor(rounded / 60)}m ${String(rounded % 60).padStart(2, '0')}s`;
  };
  const status = (message, kind = '') => {
    const node = $('validation-status');
    if (!node) return;
    node.textContent = message;
    node.className = `validation-status ${kind}`;
  };
  const mark = value => value === true
    ? '<span class="good">✓</span>'
    : value === false
      ? '<span class="bad">✕</span>'
      : '<span class="metric-unavailable">—</span>';

  let runs = [];
  let selectedRun = null;
  let telemetry = [];
  let selectedStage = STAGES[0];
  let selectedMetric = null;
  let selectedRange = null;
  let telemetryChart = null;
  let selectionToken = 0;
  let chartDragStart = null;

  const resizeHandle = $("selected-run-resize");
  const selectedRunPanel = $("run-details");
  const LEFT_MIN_RATIO = 0.35;
  const RIGHT_MIN_RATIO = 0.30;
  let resizePointerId = null;

  const resizeAnalysisArea = clientX => {
    const bounds = selectedRunPanel.getBoundingClientRect();
    const available = selectedRunPanel.clientWidth - 20 - resizeHandle.offsetWidth;
    const minimum = available * LEFT_MIN_RATIO;
    const maximum = available * (1 - RIGHT_MIN_RATIO);
    const width = Math.max(minimum, Math.min(maximum, clientX - bounds.left - 10));
    selectedRunPanel.style.setProperty("--validation-left-width", width + "px");
    if (telemetryChart) telemetryChart.resize();
  };

  resizeHandle.addEventListener("pointerdown", event => {
    resizePointerId = event.pointerId;
    resizeHandle.setPointerCapture(event.pointerId);
    resizeHandle.classList.add("dragging");
    resizeAnalysisArea(event.clientX);
  });
  resizeHandle.addEventListener("pointermove", event => {
    if (event.pointerId === resizePointerId) resizeAnalysisArea(event.clientX);
  });
  const finishResize = event => {
    if (event.pointerId !== resizePointerId) return;
    resizePointerId = null;
    resizeHandle.classList.remove("dragging");
  };
  resizeHandle.addEventListener("pointerup", finishResize);
  resizeHandle.addEventListener("pointercancel", finishResize);

  const stageForRow = row => {
    const state = String(row.state || '').toUpperCase();
    if (stateToStage.has(state)) return stateToStage.get(state);
    if (state === 'SUCCESS') {
      const dockingValue = [row.dock_geometry_distance_m, row.dock_lateral_error_m, row.dock_insertion_depth_m]
        .some(value => finite(value) !== null);
      return dockingValue || row.phase === 'DOCKING' || row.is_docked === true ? 6 : 3;
    }
    return null;
  };

  const stageRows = stage => telemetry.filter(row => stageForRow(row) === stage.index && timeOf(row) !== null);
  const availableMetrics = (stage, rows = stageRows(stage)) => stage.metrics
    .map(([field, label, unit]) => ({ field, label, unit }))
    .filter(metric => rows.some(row => finite(row[metric.field]) !== null));

  const setPlaceholder = message => {
    $('validation-telemetry-chart').hidden = true;
    $('telemetry-placeholder').hidden = false;
    $('telemetry-placeholder').textContent = message;
    $('range-controls').hidden = true;
    $('export-json').disabled = true;
    selectedRange = null;
    updateSelectedData();
  };

  const renderRunInfo = (run, recordCount = null) => {
    if (!run) {
      $('run-info').innerHTML = '<span class="metric-unavailable">Select a run</span>';
      return;
    }
    const rows = [
      ['Mission Result', run.missionSuccess === true ? 'SUCCESS' : run.missionSuccess === false ? 'FAILED' : 'INCOMPLETE / N/A'],
      ['Start Time (UTC)', date(run.createdAt)],
      ['Total Time', formatDuration(run.duration)],
      ['Final State', text(run.finalState)],
      ['Telemetry Records', recordCount === null ? 'Loading…' : String(recordCount)],
      ['Capture GT Error', number(run.captureError, ' m')],
      ['Docking Error', number(run.dockingError, ' m')]
    ];
    if (run.failureStage) rows.push(['Failure Stage', text(run.failureStage)]);
    if (run.failureReason) rows.push(['Failure Reason', text(run.failureReason)]);
    $('run-info').innerHTML = `<dl>${rows.map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
  };

  const renderHistory = list => {
    $('run-history').innerHTML = [...list].reverse().map((run, index) =>
      `<tr data-session="${esc(run.sessionId)}"><td>${list.length - index}</td><td><button aria-label="Select ${esc(run.sessionId)}" aria-pressed="false">${esc(run.sessionId)}</button></td><td>${mark(run.missionSuccess)}</td><td>${number(run.duration, '', 1)}</td><td>${esc(date(run.createdAt))}</td></tr>`
    ).join('');
    $('run-history').onclick = event => {
      const row = event.target.closest('[data-session]');
      if (!row) return;
      const run = list.find(item => item.sessionId === row.dataset.session);
      if (run) selectRun(run, row);
    };
  };

  // Session video timing: video time 0 = sim time `videoT0` (null: legacy video, no range sync)
  let videoT0 = null;

  // Selected graph range [sim s] -> video time [s], clamped to the video
  const videoRange = () => {
    const video = $('run-video');
    if (!selectedRange || videoT0 === null || video.hidden) return null;
    const duration = Number.isFinite(video.duration) ? video.duration : Infinity;
    const clamp = value => Math.min(Math.max(value, 0), duration);
    return { start: clamp(selectedRange.start - videoT0), end: clamp(selectedRange.end - videoT0) };
  };

  // Show the first frame of the selected range (paused)
  const seekVideoToRange = () => {
    const video = $('run-video');
    const range = videoRange();
    if (!range || video.readyState < 1) return;
    video.pause();
    try { video.currentTime = range.start; } catch (_) { /* metadata not ready */ }
  };

  const resetVideo = () => {
    const video = $('run-video');
    videoT0 = null;
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.hidden = true;
    $('video-placeholder').hidden = false;
    $('video-placeholder').querySelector('span').textContent = 'VIDEO NOT AVAILABLE';
    $('video-status').textContent = 'CHECKING VIDEO…';
    $('video-play').disabled = true;
    $('video-stop').disabled = true;
  };

  const loadVideo = async (run, token) => {
    resetVideo();
    try {
      const metadata = await window.ValidationFirestore.loadVideoMetadata(run.sessionId);
      if (token !== selectionToken) return;
      if (!metadata.available || !metadata.url) {
        $('video-status').textContent = 'VIDEO NOT AVAILABLE';
        return;
      }
      const video = $('run-video');
      videoT0 = Number.isFinite(metadata.video_t0_s) ? metadata.video_t0_s : null;
      video.onloadedmetadata = seekVideoToRange;
      video.src = metadata.url;
      video.hidden = false;
      $('video-placeholder').hidden = true;
      $('video-status').textContent = `${metadata.size_bytes ? (metadata.size_bytes / 1048576).toFixed(1) + ' MB' : 'VIDEO AVAILABLE'}`
        + (videoT0 === null ? ' · NO TIME SYNC' : ' · RANGE SYNC');
      $('video-play').disabled = false;
      $('video-stop').disabled = false;
      video.load();
    } catch (_) {
      if (token === selectionToken) $('video-status').textContent = 'VIDEO NOT AVAILABLE';
    }
  };

  // PLAY plays the selected graph range (from its start to its end); without timing the whole video
  // The PLAY button turns into PAUSE while the video plays (and back when it stops)
  const updatePlayButton = () => {
    $('video-play').textContent = $('run-video').paused ? '▶ PLAY' : '❚❚ PAUSE';
  };
  $('run-video').onplay = updatePlayButton;
  $('run-video').onpause = updatePlayButton;
  $('run-video').onended = updatePlayButton;
  $('run-video').onemptied = updatePlayButton;

  $('video-play').onclick = () => {
    const video = $('run-video');
    if (video.hidden) return;
    if (!video.paused) {
      video.pause();
      return;
    }
    const range = videoRange();
    if (range && (video.currentTime < range.start - 0.05 || video.currentTime >= range.end - 0.05)) {
      try { video.currentTime = range.start; } catch (_) { /* metadata not ready */ }
    }
    video.play().catch(() => {});
  };
  $('video-stop').onclick = () => {
    const video = $('run-video');
    video.pause();
    const range = videoRange();
    try { video.currentTime = range ? range.start : 0; } catch (_) { /* metadata not ready */ }
  };
  $('run-video').ontimeupdate = () => {
    const video = $('run-video');
    const range = videoRange();
    if (range && !video.paused && video.currentTime >= range.end) {
      video.pause();
      try { video.currentTime = range.end; } catch (_) { /* metadata not ready */ }
    }
  };

  const renderStageSelector = () => {
    $('stage-selector').innerHTML = STAGES.map(stage => {
      const count = stageRows(stage).length;
      return `<button role="tab" data-stage="${stage.index}" aria-selected="${stage.index === selectedStage.index}" class="${stage.index === selectedStage.index ? 'active' : ''}"><b>${stage.index}</b><span>${esc(stage.label)}</span><small>${count ? `${count} samples` : 'NO DATA'}</small></button>`;
    }).join('');
    $('stage-selector').onclick = event => {
      const button = event.target.closest('[data-stage]');
      if (!button) return;
      selectedStage = STAGES.find(stage => stage.index === Number(button.dataset.stage));
      selectedMetric = null;
      renderStageSelector();
      renderMetricSelector();
      renderTelemetryChart();
    };
  };

  const renderMetricSelector = () => {
    const metrics = availableMetrics(selectedStage);
    if (!selectedMetric || !metrics.some(metric => metric.field === selectedMetric.field)) {
      selectedMetric = metrics[0] || null;
    }
    $('metric-selector').innerHTML = metrics.map(metric =>
      `<button data-metric="${esc(metric.field)}" class="${selectedMetric && metric.field === selectedMetric.field ? 'active' : ''}">${esc(metric.label)}</button>`
    ).join('');
    $('metric-selector').onclick = event => {
      const button = event.target.closest('[data-metric]');
      if (!button) return;
      selectedMetric = metrics.find(metric => metric.field === button.dataset.metric) || null;
      renderMetricSelector();
      renderTelemetryChart();
    };
  };

  const rangeRows = () => {
    if (!selectedRange) return [];
    return stageRows(selectedStage).filter(row => {
      const time = timeOf(row);
      return time !== null && time >= selectedRange.start && time <= selectedRange.end;
    });
  };

  function updateSelectedData() {
    if (!selectedRange || !selectedMetric) {
      $('selected-start').textContent = '—';
      $('selected-end').textContent = '—';
      $('selected-duration').textContent = '—';
      $('selected-stats').innerHTML = '<span class="metric-unavailable">NO DATA</span>';
      $('export-json').disabled = true;
      return;
    }

    $('selected-start').textContent = `${selectedRange.start.toFixed(2)} s`;
    $('selected-end').textContent = `${selectedRange.end.toFixed(2)} s`;
    $('selected-duration').textContent = `${Math.max(0, selectedRange.end - selectedRange.start).toFixed(2)} s`;
    $('range-start-label').textContent = `${selectedRange.start.toFixed(2)} s`;
    $('range-end-label').textContent = `${selectedRange.end.toFixed(2)} s`;

    const rows = rangeRows();
    const metricValues = rows.map(row => finite(row[selectedMetric.field])).filter(value => value !== null);
    if (!metricValues.length) {
      $('selected-stats').innerHTML = '<span class="metric-unavailable">NO DATA IN SELECTED RANGE</span>';
      $('export-json').disabled = rows.length === 0;
      return;
    }

    const stats = [
      ['Metric', selectedMetric.label],
      ['Mean', number(mean(metricValues), selectedMetric.unit)],
      ['Min', number(Math.min(...metricValues), selectedMetric.unit)],
      ['Max', number(Math.max(...metricValues), selectedMetric.unit)],
      ['Latest / End', number(metricValues[metricValues.length - 1], selectedMetric.unit)],
      ['Sample Count', String(metricValues.length)]
    ];
    $('selected-stats').innerHTML = `<dl>${stats.map(([key, value]) => `<div><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl>`;
    $('export-json').disabled = false;
  }

  const setRange = (start, end, syncInputs = true) => {
    if (!Number.isFinite(start) || !Number.isFinite(end)) return;
    selectedRange = { start: Math.min(start, end), end: Math.max(start, end) };
    if (syncInputs) {
      $('range-start').value = String(selectedRange.start);
      $('range-end').value = String(selectedRange.end);
    }
    updateSelectedData();
    seekVideoToRange();
    if (telemetryChart) telemetryChart.draw();
  };

  const selectionPlugin = {
    id: 'validationRangeSelection',
    afterDatasetsDraw(chart) {
      if (!selectedRange || !chart.scales.x) return;
      const left = chart.scales.x.getPixelForValue(selectedRange.start);
      const right = chart.scales.x.getPixelForValue(selectedRange.end);
      const { top, bottom } = chart.chartArea;
      const ctx = chart.ctx;
      ctx.save();
      ctx.fillStyle = 'rgba(137, 229, 252, .12)';
      ctx.strokeStyle = 'rgba(137, 229, 252, .9)';
      ctx.lineWidth = 1;
      ctx.fillRect(left, top, right - left, bottom - top);
      ctx.strokeRect(left, top, right - left, bottom - top);
      ctx.restore();
    }
  };

  const renderTelemetryChart = () => {
    const rows = stageRows(selectedStage);
    if (!rows.length) {
      setPlaceholder(selectedStage.index === 7 ? 'ORBIT TRANSFER TELEMETRY NOT IMPLEMENTED' : 'NO TELEMETRY FOR THIS STAGE');
      $('metric-selector').innerHTML = '';
      return;
    }
    if (!selectedMetric) {
      setPlaceholder('NO MEASURED METRIC FOR THIS STAGE');
      return;
    }

    const points = rows.map(row => ({ x: timeOf(row), y: finite(row[selectedMetric.field]) }))
      .filter(point => point.x !== null && point.y !== null)
      .sort((a, b) => a.x - b.x);
    if (!points.length) {
      setPlaceholder('NO DATA FOR SELECTED METRIC');
      return;
    }

    const minTime = points[0].x;
    const maxTime = points[points.length - 1].x;
    const inputStep = Math.max((maxTime - minTime) / 1000, 0.001);
    ['range-start', 'range-end'].forEach(id => {
      $(id).min = String(minTime);
      $(id).max = String(maxTime);
      $(id).step = String(inputStep);
    });
    $('range-controls').hidden = false;
    $('validation-telemetry-chart').hidden = false;
    $('telemetry-placeholder').hidden = true;
    setRange(minTime, maxTime);

    if (telemetryChart) telemetryChart.destroy();
    telemetryChart = new Chart($('validation-telemetry-chart'), {
      type: 'line',
      data: {
        datasets: [{
          label: `${selectedMetric.label}${selectedMetric.unit}`,
          data: points,
          borderColor: '#89e5fc',
          backgroundColor: 'rgba(137, 229, 252, .08)',
          borderWidth: 2,
          pointRadius: points.length > 250 ? 0 : 1.5,
          tension: .12,
          fill: false
        }]
      },
      plugins: [selectionPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        parsing: false,
        normalized: true,
        plugins: { legend: { display: false } },
        scales: {
          x: Dashboard.axis('Simulation Time [s]', { type: 'linear', min: minTime, max: maxTime }),
          y: Dashboard.axis(`${selectedMetric.label}${selectedMetric.unit}`)
        }
      }
    });
    Dashboard.charts.push(telemetryChart);
  };

  $('range-start').oninput = () => setRange(Number($('range-start').value), Number($('range-end').value));
  $('range-end').oninput = () => setRange(Number($('range-start').value), Number($('range-end').value));

  const chartCanvas = $('validation-telemetry-chart');
  const chartTimeAt = event => {
    if (!telemetryChart || !telemetryChart.chartArea) return null;
    const rect = chartCanvas.getBoundingClientRect();
    const pixel = Math.max(telemetryChart.chartArea.left, Math.min(telemetryChart.chartArea.right, event.clientX - rect.left));
    const value = telemetryChart.scales.x.getValueForPixel(pixel);
    return Number.isFinite(value) ? value : null;
  };
  chartCanvas.addEventListener('pointerdown', event => {
    const value = chartTimeAt(event);
    if (value === null) return;
    chartDragStart = value;
    chartCanvas.setPointerCapture(event.pointerId);
    setRange(value, value);
  });
  chartCanvas.addEventListener('pointermove', event => {
    if (chartDragStart === null) return;
    const value = chartTimeAt(event);
    if (value !== null) setRange(chartDragStart, value);
  });
  const finishChartDrag = event => {
    if (chartDragStart === null) return;
    const value = chartTimeAt(event);
    if (value !== null) setRange(chartDragStart, value);
    chartDragStart = null;
  };
  chartCanvas.addEventListener('pointerup', finishChartDrag);
  chartCanvas.addEventListener('pointercancel', () => { chartDragStart = null; });

  $('export-json').onclick = () => {
    if (!selectedRun || !selectedMetric || !selectedRange) return;
    const payload = {
      _export: {
        run_id: selectedRun.sessionId,
        mission_stage_index: selectedStage.index,
        mission_stage_key: selectedStage.key,
        metric: selectedMetric.field,
        start_time_s: selectedRange.start,
        end_time_s: selectedRange.end,
        duration_s: selectedRange.end - selectedRange.start
      },
      simulation_session: selectedRun.raw,
      session_telemetry: rangeRows()
    };
    const filename = `${selectedRun.sessionId}_${selectedStage.key}_${selectedRange.start.toFixed(2)}s_${selectedRange.end.toFixed(2)}s.json`
      .replace(/[^A-Za-z0-9_.-]/g, '_');
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  async function selectRun(run, row = null) {
    selectedRun = run;
    telemetry = [];
    selectedStage = STAGES[0];
    selectedMetric = null;
    selectedRange = null;
    const token = ++selectionToken;

    document.querySelectorAll('#run-history [data-session]').forEach(item => {
      const selected = row ? item === row : item.dataset.session === run.sessionId;
      item.classList.toggle('selected', selected);
      const button = item.querySelector('button');
      if (button) button.setAttribute('aria-pressed', String(selected));
    });

    $('selected-run-id').textContent = `Run ID: ${run.sessionId}`;
    $('run-result').textContent = run.missionSuccess === true ? '✓ SUCCESS' : run.missionSuccess === false ? '✕ FAILED' : 'INCOMPLETE / N/A';
    $('run-result').className = `result ${run.missionSuccess === true ? 'good' : run.missionSuccess === false ? 'bad' : 'metric-unavailable'}`;
    renderRunInfo(run);
    renderStageSelector();
    $('metric-selector').innerHTML = '';
    setPlaceholder('LOADING TELEMETRY…');
    loadVideo(run, token);

    try {
      const response = await window.ValidationFirestore.loadTelemetry(run.sessionId);
      if (token !== selectionToken) return;
      telemetry = Array.isArray(response.telemetry) ? response.telemetry : [];
      telemetry.sort((a, b) => (timeOf(a) ?? 0) - (timeOf(b) ?? 0));
      renderRunInfo(run, telemetry.length);
      renderStageSelector();
      const firstAvailable = STAGES.find(stage => availableMetrics(stage).length);
      selectedStage = firstAvailable || STAGES[0];
      renderStageSelector();
      renderMetricSelector();
      renderTelemetryChart();
    } catch (error) {
      if (token !== selectionToken) return;
      renderRunInfo(run, 0);
      renderStageSelector();
      setPlaceholder(`TELEMETRY UNAVAILABLE · ${error.message}`);
    }
  }

  try {
    status('Loading Firestore run summaries…');
    const response = await window.ValidationFirestore.loadRuns();
    runs = (Array.isArray(response.runs) ? response.runs : [])
      .map(normalizeRun)
      .filter(run => run.sessionId)
      .sort((a, b) => String(a.createdAt || '').localeCompare(String(b.createdAt || '')));

    $('total-runs').textContent = String(runs.length);
    $('total-runs-note').textContent = `${runs.length} Firestore session records`;

    if (!runs.length) {
      status('NO FIRESTORE RUN DATA', 'empty');
      $('overall-rate').textContent = 'N/A';
      $('overall-count').textContent = 'No valid mission_success';
      $('avg-time').textContent = 'N/A';
      $('capture-repeatability').textContent = 'N/A';
      $('capture-repeatability-note').textContent = 'No GT capture metric';
      $('docking-accuracy').textContent = 'N/A';
      $('docking-accuracy-note').textContent = 'No docking metric';
      $('run-history').innerHTML = '<tr><td colspan="5">NO FIRESTORE RUN DATA</td></tr>';
      setPlaceholder('NO FIRESTORE RUN DATA');
      return;
    }

    const overall = successRate(runs.map(run => run.missionSuccess));
    const duration = values(runs.filter(fullPipelineSuccess), 'duration');
    const averageDuration = mean(duration);
    $('overall-rate').textContent = percent(overall);
    $('overall-count').textContent = overall ? `${overall.count} / ${overall.total} mission_success` : 'N/A — mission_success unavailable';
    $('avg-time').textContent = formatDuration(averageDuration);

    const captureSpread = spread(values(runs.filter(run => run.captureSuccess === true), 'captureError'));
    const dockingSpread = spread(values(runs.filter(run => run.dockingSuccess === true), 'dockingError'));
    $('capture-repeatability').textContent = formatSpread(captureSpread);
    $('capture-repeatability-note').textContent = captureSpread ? spreadNote(captureSpread) : 'No GT capture metric';
    $('docking-accuracy').textContent = formatSpread(dockingSpread);
    $('docking-accuracy-note').textContent = dockingSpread ? spreadNote(dockingSpread) : 'No docking metric';

    renderHistory(runs);
    const latest = runs[runs.length - 1];
    selectRun(latest);

    const gtCaptureRuns = runs.filter(run => run.hasGtCaptureMetric).length;
    // `cache: stale` means Firestore was unreachable or out of quota and the
    // server fell back to its last copy -- say so rather than look up to date.
    const stale = response.cache === 'stale';
    status(`TECHNOLOGY VALIDATION · ${stale ? 'CACHED DATA (FIRESTORE UNAVAILABLE)' : 'FIRESTORE DATA'}`
      + ` · ${runs.length} RUNS · ${gtCaptureRuns} GT CAPTURE METRIC RUN${gtCaptureRuns === 1 ? '' : 'S'}`
      + (stale ? ` · ${response.cache_error || ''}` : ''), stale ? 'warn' : '');
  } catch (error) {
    status(`FIREBASE CONNECTION ERROR · ${error.message}`, 'error');
    ['overall-rate', 'avg-time', 'capture-repeatability', 'docking-accuracy'].forEach(id => { $(id).textContent = 'N/A'; });
    $('overall-count').textContent = 'Firebase Connection Error';
    $('total-runs').textContent = '—';
    $('total-runs-note').textContent = 'Firebase Connection Error';
    $('capture-repeatability-note').textContent = 'Firebase Connection Error';
    $('docking-accuracy-note').textContent = 'Firebase Connection Error';
    $('run-history').innerHTML = '<tr><td colspan="5">FIREBASE CONNECTION ERROR</td></tr>';
    setPlaceholder('FIREBASE CONNECTION ERROR');
  }
};
