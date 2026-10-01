const colors = ["#6ee7b7", "#53c6d7", "#f6c66a", "#f27c7c", "#a78bfa", "#8bc5ff"];
const form = document.getElementById("controlForm");
const runButton = document.getElementById("runButton");
const statusText = document.getElementById("statusText");
const liveDot = document.querySelector(".live-dot");
const toast = document.getElementById("toast");
let presets = {};
let spatialViewer = null;
const spatialSources = {
  thruster: {label:"推进器", color:"#6ee7b7"},
  dc_dc: {label:"DC/DC", color:"#53c6d7"},
  actuator: {label:"作动器", color:"#f6c66a"},
  electronics: {label:"电子负载", color:"#f27c7c"}
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4500);
}

function setBusy(busy) {
  runButton.disabled = busy;
  runButton.querySelector("span").textContent = busy ? "正在标定与航行仿真…" : "运行完整仿真";
  statusText.textContent = busy ? "计算进行中" : "系统就绪";
  liveDot.classList.toggle("running", busy);
}

function setDefaults(values) {
  Object.entries(values).forEach(([key, value]) => {
    const input = form.elements[key];
    if (!input) return;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = input.dataset.percent !== undefined ? Number(value) * 100 : value;
  });
}

function readConfig() {
  const config = {};
  [...form.elements].forEach(input => {
    if (!input.name) return;
    if (input.type === "checkbox") config[input.name] = input.checked;
    else if (input.type === "number") {
      let value = Number(input.value);
      if (input.dataset.percent !== undefined) value /= 100;
      config[input.name] = value;
    } else config[input.name] = input.value;
  });
  return config;
}

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) return;
  setDefaults(preset.values);
  form.elements.scenario_preset.value = name;
  document.querySelectorAll("[data-preset]").forEach(button => {
    button.classList.toggle("active", button.dataset.preset === name);
  });
  document.getElementById("presetDescription").textContent = preset.description;
}

document.querySelectorAll("[data-preset]").forEach(button => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

form.addEventListener("input", event => {
  if (event.target.name === "scenario_preset") return;
  form.elements.scenario_preset.value = "custom";
  document.querySelectorAll("[data-preset]").forEach(button => button.classList.remove("active"));
  document.getElementById("presetDescription").textContent = "自定义参数：当前设置不再与预设完全一致。";
});

function niceRange(min, max) {
  if (!isFinite(min) || !isFinite(max)) return [-1, 1];
  if (min === max) return [min - 1, max + 1];
  const margin = (max - min) * .1;
  return [min - margin, max + margin];
}

function drawChart(canvasId, legendId, plot, options = {}) {
  const canvas = document.getElementById(canvasId);
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(400, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  const pad = {left: 49, right: 13, top: 12, bottom: 31};
  const names = Object.keys(plot.series);
  const visibleNames = options.include || names;
  const data = visibleNames.flatMap(name => plot.series[name]);
  let [yMin, yMax] = niceRange(Math.min(...data), Math.max(...data));
  if (options.zeroCentered) { const a = Math.max(Math.abs(yMin), Math.abs(yMax)); yMin = -a; yMax = a; }
  if (options.nonNegative) yMin = 0;
  const xMin = Math.min(...plot.x), xMax = Math.max(...plot.x);
  const px = x => pad.left + (x - xMin) / Math.max(xMax - xMin, 1e-12) * (width - pad.left - pad.right);
  const py = y => height - pad.bottom - (y - yMin) / Math.max(yMax - yMin, 1e-12) * (height - pad.top - pad.bottom);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "9px Consolas";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (height - pad.top - pad.bottom) * i / 4;
    ctx.strokeStyle = "rgba(161,196,202,.12)"; ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width-pad.right, y); ctx.stroke();
    const value = yMax - (yMax-yMin)*i/4;
    ctx.fillStyle = "#6f898d"; ctx.textAlign = "right"; ctx.fillText(value.toFixed(Math.abs(value)<10?2:1), pad.left-6, y+3);
  }
  for (let i = 0; i <= 5; i++) {
    const x = pad.left + (width-pad.left-pad.right)*i/5;
    ctx.fillStyle = "#6f898d"; ctx.textAlign = "center"; ctx.fillText((xMin+(xMax-xMin)*i/5).toFixed(1), x, height-9);
  }
  visibleNames.forEach((name, seriesIndex) => {
    const values = plot.series[name];
    const style = options.styles?.[name] || {};
    ctx.strokeStyle = style.color || colors[seriesIndex % colors.length];
    ctx.lineWidth = style.width || (name.includes("ideal") ? 2.2 : 1.35);
    ctx.setLineDash(style.dash || (name.includes("predicted") || name.includes("ideal") ? [5, 4] : []));
    ctx.beginPath();
    values.forEach((value, i) => { const x = px(plot.x[i]), y = py(value); if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke(); ctx.setLineDash([]);
  });
  ctx.fillStyle = "#6f898d"; ctx.textAlign="right"; ctx.fillText(options.xLabel || "时间 / s", width-pad.right, height-9);
  const legend = document.getElementById(legendId);
  legend.innerHTML = visibleNames.map((name,i) => `<span><i style="background:${options.styles?.[name]?.color || colors[i%colors.length]}"></i>${options.labels?.[name] || name}</span>`).join("");
}

function vectorMagnitude(vector) {
  return Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
}

function initSpatialViewer(scene) {
  if (spatialViewer) spatialViewer.destroy();
  const canvas = document.getElementById("auv3dCanvas");
  const timeline = document.getElementById("spatialTimeline");
  const playButton = document.getElementById("spatialPlayButton");
  const timeLabel = document.getElementById("spatialTime");
  const sourceList = document.getElementById("sourceSpatialList");
  const sourceNames = Object.keys(spatialSources);
  const frameCount = scene.time_s.length;
  let frame = 0;
  let playing = true;
  let yaw = -0.48;
  let pitch = 0.34;
  let zoom = 1.0;
  let dragging = false;
  let dragX = 0;
  let dragY = 0;
  let animationFrame = 0;
  let lastStep = performance.now();

  timeline.max = Math.max(frameCount - 1, 0);
  timeline.value = 0;
  sourceList.innerHTML = sourceNames.map(name => {
    const meta = spatialSources[name];
    const position = scene.source_positions_m[name];
    const distance = vectorMagnitude(position);
    return `<div class="spatial-source-row" style="--source-color:${meta.color}">
      <header><b>${meta.label}</b><span id="scene-${name}-field">— nT</span></header>
      <p><span id="scene-${name}-state">s = —</span><span>r = ${distance.toFixed(3)} m</span></p>
    </div>`;
  }).join("");

  function fitCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width));
    const height = Math.max(280, Math.floor(rect.height));
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {ctx, width, height};
  }

  function project(point, width, height) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = cy * point[0] - sy * point[1];
    const y1 = sy * point[0] + cy * point[1];
    const z1 = point[2];
    const y2 = cp * y1 - sp * z1;
    const z2 = sp * y1 + cp * z1;
    const extent = Math.max(1.55, ...sourceNames.map(name => vectorMagnitude(scene.source_positions_m[name]) * 1.15));
    const scale = Math.min(width / (2.45 * extent), height / (1.65 * extent)) * zoom;
    const perspective = 1 / Math.max(0.68, 1 + 0.13 * y2);
    return {
      x: width * 0.50 + x1 * scale * perspective,
      y: height * 0.48 - z2 * scale * perspective,
      depth: y2,
      scale: scale * perspective
    };
  }

  function drawPolyline(ctx, points, width, height, stroke, lineWidth=1, close=false) {
    const projected = points.map(point => project(point, width, height));
    ctx.beginPath();
    projected.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
    if (close) ctx.closePath();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    return projected;
  }

  function drawGrid(ctx, width, height) {
    ctx.lineWidth = 1;
    for (let x = -1.5; x <= 1.5; x += 0.25) {
      drawPolyline(ctx, [[x,-0.75,-0.38],[x,0.75,-0.38]], width, height, "rgba(83,198,215,.065)");
    }
    for (let y = -0.75; y <= 0.75; y += 0.25) {
      drawPolyline(ctx, [[-1.5,y,-0.38],[1.5,y,-0.38]], width, height, "rgba(83,198,215,.065)");
    }
  }

  function drawBody(ctx, width, height) {
    const rings = 18;
    const segments = 18;
    const vertices = [];
    for (let ring = 0; ring <= rings; ring++) {
      const u = ring / rings;
      const x = -1.32 + 2.34 * u;
      const radius = 0.035 + 0.31 * Math.pow(Math.sin(Math.PI * u), 0.58);
      const row = [];
      for (let segment = 0; segment < segments; segment++) {
        const angle = 2 * Math.PI * segment / segments;
        row.push([x, radius * Math.cos(angle), radius * Math.sin(angle)]);
      }
      vertices.push(row);
    }
    const faces = [];
    for (let ring = 0; ring < rings; ring++) {
      for (let segment = 0; segment < segments; segment++) {
        const next = (segment + 1) % segments;
        const a = project(vertices[ring][segment], width, height);
        const b = project(vertices[ring + 1][segment], width, height);
        const c = project(vertices[ring + 1][next], width, height);
        const d = project(vertices[ring][next], width, height);
        faces.push({points:[a,b,c,d], depth:(a.depth+b.depth+c.depth+d.depth)/4, shade:segment/segments});
      }
    }
    faces.sort((left,right) => left.depth-right.depth);
    faces.forEach(face => {
      const alpha = 0.075 + 0.055 * (1 + Math.sin(face.shade * 2 * Math.PI));
      ctx.beginPath();
      face.points.forEach((point,index) => index ? ctx.lineTo(point.x,point.y) : ctx.moveTo(point.x,point.y));
      ctx.closePath();
      ctx.fillStyle = `rgba(36,128,139,${alpha})`;
      ctx.fill();
      ctx.strokeStyle = "rgba(83,198,215,.10)";
      ctx.lineWidth = .55;
      ctx.stroke();
    });

    const finSets = [
      [[-0.93,0,0.18],[-1.20,0,0.60],[-1.28,0,0.17]],
      [[-0.93,0,-0.18],[-1.20,0,-0.60],[-1.28,0,-0.17]],
      [[-0.93,0.18,0],[-1.20,0.62,0],[-1.28,0.17,0]],
      [[-0.93,-0.18,0],[-1.20,-0.62,0],[-1.28,-0.17,0]]
    ];
    finSets.forEach(fin => {
      const points = fin.map(point => project(point,width,height));
      ctx.beginPath();
      points.forEach((point,index) => index ? ctx.lineTo(point.x,point.y) : ctx.moveTo(point.x,point.y));
      ctx.closePath();
      ctx.fillStyle = "rgba(65,145,153,.22)";
      ctx.fill();
      ctx.strokeStyle = "rgba(83,198,215,.32)";
      ctx.stroke();
    });

    const propeller = [];
    for (let index=0; index<=32; index++) {
      const angle = 2*Math.PI*index/32;
      propeller.push([-1.39, .39*Math.cos(angle), .39*Math.sin(angle)]);
    }
    drawPolyline(ctx, propeller, width, height, "rgba(110,231,183,.45)", 1.2, true);
  }

  function drawArrow(ctx, start3d, vector, maxMagnitude, width, height, color, label) {
    const magnitude = vectorMagnitude(vector);
    if (magnitude < 1e-9) return;
    const unit = vector.map(value => value / magnitude);
    const length = 0.13 + 0.68 * Math.sqrt(magnitude / Math.max(maxMagnitude, 1e-9));
    const end3d = start3d.map((value,index) => value + unit[index] * length);
    const start = project(start3d,width,height);
    const end = project(end3d,width,height);
    const angle = Math.atan2(end.y-start.y,end.x-start.x);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.1;
    ctx.beginPath(); ctx.moveTo(start.x,start.y); ctx.lineTo(end.x,end.y); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(end.x,end.y);
    ctx.lineTo(end.x-9*Math.cos(angle-.42),end.y-9*Math.sin(angle-.42));
    ctx.lineTo(end.x-9*Math.cos(angle+.42),end.y-9*Math.sin(angle+.42));
    ctx.closePath(); ctx.fill();
    ctx.font = "10px Consolas";
    ctx.fillText(label, end.x+6, end.y-5);
  }

  function drawFrame(now) {
    const {ctx,width,height} = fitCanvas();
    ctx.clearRect(0,0,width,height);
    drawGrid(ctx,width,height);
    drawBody(ctx,width,height);

    const sensorPoint = project([0,0,0],width,height);
    const frameSourceMagnitudes = sourceNames.map(name => vectorMagnitude(scene.source_field_vectors_nt[name][frame]));
    const maxSourceMagnitude = Math.max(...frameSourceMagnitudes,1e-9);
    sourceNames.forEach((name,index) => {
      const meta = spatialSources[name];
      const position = scene.source_positions_m[name];
      const sourcePoint = project(position,width,height);
      const magnitude = frameSourceMagnitudes[index];
      const intensity = Math.sqrt(magnitude/maxSourceMagnitude);
      ctx.save();
      ctx.setLineDash([4,5]);
      ctx.strokeStyle = meta.color;
      ctx.globalAlpha = .25 + .58*intensity;
      ctx.lineWidth = 1 + 2.1*intensity;
      ctx.beginPath(); ctx.moveTo(sourcePoint.x,sourcePoint.y); ctx.lineTo(sensorPoint.x,sensorPoint.y); ctx.stroke();
      ctx.setLineDash([]);
      const travel = (now/1800 + index*.23) % 1;
      const particleX = sourcePoint.x + (sensorPoint.x-sourcePoint.x)*travel;
      const particleY = sourcePoint.y + (sensorPoint.y-sourcePoint.y)*travel;
      ctx.beginPath(); ctx.arc(particleX,particleY,2.4+2*intensity,0,2*Math.PI); ctx.fillStyle=meta.color; ctx.fill();
      const state = Math.abs(scene.source_states[name][frame]);
      ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(sourcePoint.x,sourcePoint.y,5+5*Math.min(state,1.1),0,2*Math.PI);
      ctx.fillStyle = meta.color; ctx.fill();
      ctx.strokeStyle = "rgba(232,242,241,.75)"; ctx.lineWidth=1; ctx.stroke();
      ctx.font = "10px Microsoft YaHei"; ctx.fillStyle="#cfe0e1";
      ctx.fillText(meta.label,sourcePoint.x+9,sourcePoint.y-8);
      ctx.restore();
    });

    ctx.beginPath(); ctx.arc(sensorPoint.x,sensorPoint.y,13+2*Math.sin(now/280),0,2*Math.PI);
    ctx.fillStyle="rgba(167,139,250,.16)"; ctx.fill();
    ctx.beginPath(); ctx.arc(sensorPoint.x,sensorPoint.y,5,0,2*Math.PI);
    ctx.fillStyle="#a78bfa"; ctx.fill();
    ctx.font="10px Microsoft YaHei"; ctx.fillStyle="#d9ccff";
    ctx.fillText("铯光泵",sensorPoint.x+10,sensorPoint.y+17);

    const raw = scene.true_total_vector_nt[frame];
    const coil = scene.coil_vector_nt[frame];
    const residual = scene.residual_vector_nt[frame];
    const maxVector = Math.max(vectorMagnitude(raw),vectorMagnitude(coil),vectorMagnitude(residual),1e-9);
    drawArrow(ctx,[0,0,0],raw,maxVector,width,height,"#f27c7c","自磁");
    drawArrow(ctx,[0,0,0],coil,maxVector,width,height,"#f6c66a","线圈");
    drawArrow(ctx,[0,0,0],residual,maxVector,width,height,"#6ee7b7","残差");
  }

  function updateReadout() {
    timeLabel.textContent = `t = ${scene.time_s[frame].toFixed(2)} s`;
    timeline.value = frame;
    sourceNames.forEach(name => {
      const state = scene.source_states[name][frame];
      const magnitude = vectorMagnitude(scene.source_field_vectors_nt[name][frame]);
      document.getElementById(`scene-${name}-state`).textContent = `s = ${state.toFixed(3)}`;
      document.getElementById(`scene-${name}-field`).textContent = `${magnitude.toFixed(2)} nT`;
    });
    document.getElementById("spatialRawField").textContent = vectorMagnitude(scene.true_total_vector_nt[frame]).toFixed(2);
    document.getElementById("spatialPredictedField").textContent = vectorMagnitude(scene.predicted_total_vector_nt[frame]).toFixed(2);
    document.getElementById("spatialResidualField").textContent = vectorMagnitude(scene.residual_vector_nt[frame]).toFixed(3);
    const sensorError = scene.stabilized_sensor_error_nt[frame];
    document.getElementById("spatialSensorError").textContent = `${sensorError >= 0 ? "+" : ""}${sensorError.toFixed(3)}`;
  }

  function tick(now) {
    if (playing && frameCount > 1 && now-lastStep >= 55) {
      const steps = Math.max(1,Math.floor((now-lastStep)/55));
      frame = (frame+steps)%frameCount;
      lastStep = now;
      updateReadout();
    }
    drawFrame(now);
    animationFrame = requestAnimationFrame(tick);
  }

  playButton.onclick = () => {
    playing = !playing;
    playButton.textContent = playing ? "暂停" : "播放";
    lastStep = performance.now();
  };
  timeline.oninput = () => {
    frame = Number(timeline.value);
    playing = false;
    playButton.textContent = "播放";
    updateReadout();
  };
  canvas.onpointerdown = event => {
    dragging = true; dragX=event.clientX; dragY=event.clientY; canvas.setPointerCapture(event.pointerId);
  };
  canvas.onpointermove = event => {
    if (!dragging) return;
    yaw += (event.clientX-dragX)*.008;
    pitch = Math.max(-1.05,Math.min(1.05,pitch+(event.clientY-dragY)*.006));
    dragX=event.clientX; dragY=event.clientY;
  };
  canvas.onpointerup = () => { dragging=false; };
  canvas.onpointercancel = () => { dragging=false; };
  canvas.onwheel = event => {
    event.preventDefault();
    zoom = Math.max(.68,Math.min(1.8,zoom*(event.deltaY>0?.92:1.08)));
  };

  updateReadout();
  animationFrame = requestAnimationFrame(tick);
  spatialViewer = {
    destroy() {
      cancelAnimationFrame(animationFrame);
      canvas.onpointerdown=canvas.onpointermove=canvas.onpointerup=canvas.onpointercancel=canvas.onwheel=null;
      playButton.onclick=timeline.oninput=null;
    }
  };
}

function fmt(value, digits=2) { return Number(value).toFixed(digits); }

function render(result) {
  const summary = result.summary;
  const joint = summary.mode_metrics.joint;
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");
  document.getElementById("runSeed").textContent = `SEED ${result.config.seed}`;
  const preset = presets[result.config.scenario_preset];
  document.getElementById("runPreset").textContent = `PRESET ${preset ? preset.label : "自定义"}`;
  document.getElementById("runTime").textContent = `${result.config.navigation_duration_s}s @ ${result.config.sample_rate_hz}Hz`;
  document.getElementById("metricSource").textContent = fmt(summary.source_holdout_field_rmse_nt, 3);
  document.getElementById("metricCoil").textContent = fmt(summary.navigation_prediction.field_vector_rmse_nt, 4);
  document.getElementById("metricField").textContent = `${fmt(joint.field_reduction_percent, 3)}%`;
  document.getElementById("metricGradient").textContent = `${fmt(joint.gxx_reduction_percent, 3)}%`;
  document.getElementById("metricResidualField").textContent = fmt(joint.field_residual_rms_nt, 3);
  const fieldGoalPassed = summary.field_accuracy_goal.passed;
  const fieldGoalStatus = document.getElementById("metricResidualFieldStatus");
  const fieldGoalCard = document.getElementById("metricResidualFieldCard");
  fieldGoalStatus.textContent = fieldGoalPassed ? "通过 < 1 nT 独立目标" : "未通过 < 1 nT 独立目标";
  fieldGoalCard.classList.toggle("goal-pass", fieldGoalPassed);
  fieldGoalCard.classList.toggle("goal-fail", !fieldGoalPassed);
  document.getElementById("metricResidualGradient").textContent = fmt(joint.gxx_residual_rms_nt_per_m, 3);
  document.getElementById("metricCesiumError").textContent = fmt(joint.cesium_error_rms_nt, 3);
  document.getElementById("metricTarget").textContent = fmt(summary.target_protection.waveform_correlation, 4);
  document.getElementById("metricCurrent").textContent = fmt(joint.peak_current_a, 3);
  const uncompensated = summary.mode_metrics.none;
  document.getElementById("metricGradientPeakBefore").textContent = fmt(uncompensated.principal_peak_nt_per_m, 0);
  document.getElementById("metricGradientPeakAfter").textContent = fmt(joint.principal_peak_nt_per_m, 0);
  document.getElementById("metricGradientAvailability").textContent = `${fmt(100 * joint.valid_fraction_by_gradient, 2)}%`;

  drawChart("benchChart", "benchLegend", result.plots.bench_source_holdout, {labels:{observed_field_norm:"观测 |B|", predicted_field_norm:"预测 |B|", observed_gxx:"观测 Gxx", predicted_gxx:"预测 Gxx"}});
  drawChart("sourceChart", "sourceLegend", result.plots.source_states, {labels:{thruster:"推进器",dc_dc:"DC/DC",actuator:"作动器",electronics:"电子负载"}});
  drawChart("modelChart", "modelLegend", result.plots.navigation_model, {labels:{true_self_field_norm:"真实 |Bself|",base_predicted_self_field_norm:"迁移前 |Bself|",predicted_self_field_norm:"航前迁移后 |Bself|"}});
  drawChart("maneuverCurrentChart", "maneuverCurrentLegend", result.plots.maneuver_current, {
    nonNegative:true,
    labels:{thruster_equivalent_current_a:"推进器等效电流", reference_20a:"20 A参考线"},
    styles:{thruster_equivalent_current_a:{color:"#6ee7b7",width:1.8},reference_20a:{color:"#f6c66a",dash:[6,4],width:1.2}}
  });
  drawChart("gradientRiskChart", "gradientRiskLegend", result.plots.gradient_risk, {
    nonNegative:true,
    labels:{uncompensated:"无补偿最大主梯度",main_gxx_only:"仅主Gxx通道",joint_compensated:"五梯度联合后",warning_656:"656 nT/m预警",tolerance_10000:"10000 nT/m容限"},
    styles:{uncompensated:{color:"#f27c7c",width:1.8},main_gxx_only:{color:"#53c6d7",width:1.4},joint_compensated:{color:"#6ee7b7",width:1.8},warning_656:{color:"#f6c66a",dash:[5,4]},tolerance_10000:{color:"#a78bfa",dash:[7,4],width:1.5}}
  });
  const riskStatus = document.getElementById("gradientRiskStatus");
  const protectedByGradient = joint.tolerance_exceed_duration_s <= 1e-9;
  riskStatus.textContent = protectedByGradient ? "补偿后全程合规" : `仍越限 ${fmt(joint.tolerance_exceed_duration_s, 2)} s`;
  riskStatus.classList.toggle("danger", !protectedByGradient);
  riskStatus.classList.toggle("safe", protectedByGradient);
  drawChart("residualChart", "residualLegend", result.plots.field_residual, {labels:{none:"无补偿",uniform:"仅均匀场",joint_main:"场+主Gxx",joint:"场+五梯度"}});
  drawChart("gradientChart", "gradientLegend", result.plots.gxx_residual, {zeroCentered:true,labels:{none:"无补偿",uniform:"仅均匀场",joint_main:"场+主Gxx",joint:"场+五梯度"}});
  drawChart("currentChart", "currentLegend", result.plots.coil_currents, {zeroCentered:true,labels:{uniform_x:"X 均匀场",uniform_y:"Y 均匀场",uniform_z:"Z 均匀场",gradient_gxx:"Gxx",gradient_gyy:"Gyy",gradient_gxy:"Gxy",gradient_gxz:"Gxz",gradient_gyz:"Gyz"}});
  drawChart("cesiumChart", "cesiumLegend", result.plots.cesium_anomaly, {labels:{none:"无补偿读数",uniform:"仅均匀场读数",joint_main:"场+主Gxx读数",joint:"五梯度联合读数",ideal_target:"理想目标波形"}});
  initSpatialViewer(result.spatial_scene);

  const manifest = result.manifest;
  const rows = [
    ["基础模型训练", manifest.base_training_scenarios.join("、")],
    ["残差修正专用", manifest.residual_correction_scenarios.join("、")],
    ["完全留出验证", manifest.holdout_validation_scenarios.join("、")],
    ["航前迁移训练", manifest.preflight_alignment_enabled ? manifest.preflight_source_training_scenarios.join("、") : "关闭"],
    ["航前迁移留出", manifest.preflight_alignment_enabled ? manifest.preflight_source_validation_scenarios.join("、") : "关闭"],
    ["航前迁移样本", manifest.preflight_alignment_enabled ? `${manifest.preflight_samples_per_scenario} / 场景；仅航前阵列` : "0"],
    ["线圈模型验证", manifest.coil_holdout],
    ["源模型形式", `${manifest.source_model_variant} · ${manifest.source_model_feature_count}维特征`],
    ["高机动训练覆盖", manifest.navigation_excitation_coverage.all_channels_covered ? "通过：幅值与变化率均在台架训练包络内" : "未通过：存在训练包络外的航行激励"],
    ["等效自磁压力倍率", `${manifest.source_strength_scale}×（待实测标定替换）`],
    ["线圈通道与能力", `3均匀场 + 5梯度；场 ${manifest.coil_authority_scale.uniform}× / 梯度 ${manifest.coil_authority_scale.gradient}×`],
    ["台架阵列构型", manifest.array_geometry],
    ["执行机构模式", manifest.idealized_hardware ? "理想化：不限流、无滞后、无模型误差/噪声" : "非理想物理执行机构"],
    ["驱动滞后补偿", result.summary.assumptions.driver_preemphasis_enabled ? "因果一阶逆预加重；不读取铯光泵" : "关闭"],
    ["台架误差模型", `${manifest.bench_error_model.white_noise_nt} nT噪声；${manifest.bench_error_model.position_sigma_mm} mm位置；${manifest.bench_error_model.axis_sigma_deg}°轴误差；${(100*manifest.bench_error_model.gain_sigma_fraction).toFixed(3)}%增益`],
    ["航行时阵列", manifest.navigation_array_installed ? "安装" : "已拆除"],
    ["航行在线改模", manifest.model_update_during_navigation ? "允许" : "禁止"],
    ["铯光泵闭环反馈", manifest.cesium_used_for_feedback ? "使用" : "不使用"]
  ];
  document.getElementById("manifestList").innerHTML = rows.map(r=>`<dt>${r[0]}</dt><dd>${r[1]}</dd>`).join("");
  document.getElementById("savePath").textContent = result.run_directory ? `数据已保存：${result.run_directory}` : "本次仅在内存计算，未写出运行数据。";
  window.scrollTo({top: 0, behavior:"smooth"});
}

form.addEventListener("submit", async event => {
  event.preventDefault(); setBusy(true);
  try {
    const response = await fetch("/api/simulate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(readConfig())});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "仿真失败");
    render(result); statusText.textContent = "仿真完成";
  } catch (error) { showToast(error.message); statusText.textContent = "运行异常"; }
  finally { runButton.disabled=false; runButton.querySelector("span").textContent="运行完整仿真"; liveDot.classList.remove("running"); }
});

Promise.all([
  fetch("/api/defaults").then(response=>response.json()),
  fetch("/api/presets").then(response=>response.json())
]).then(([defaults, loadedPresets]) => {
  presets = loadedPresets;
  setDefaults(defaults);
  applyPreset(defaults.scenario_preset || "high_maneuver");
}).catch(()=>showToast("默认参数加载失败"));
window.addEventListener("resize", () => { /* charts redraw on the next run to keep the client dependency-free */ });
