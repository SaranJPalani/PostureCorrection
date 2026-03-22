const PHASES = ["start", "mid", "end"];

const JOINT_MAP = {
  left_elbow: ["left_shoulder", "left_elbow", "left_wrist"],
  right_elbow: ["right_shoulder", "right_elbow", "right_wrist"],
  left_shoulder: ["left_elbow", "left_shoulder", "left_hip"],
  right_shoulder: ["right_elbow", "right_shoulder", "right_hip"],
  left_hip: ["left_shoulder", "left_hip", "left_knee"],
  right_hip: ["right_shoulder", "right_hip", "right_knee"],
  left_knee: ["left_hip", "left_knee", "left_ankle"],
  right_knee: ["right_hip", "right_knee", "right_ankle"],
  left_wrist: ["left_elbow", "left_wrist", "left_index"],
  right_wrist: ["right_elbow", "right_wrist", "right_index"],
  left_ankle: ["left_knee", "left_ankle", "left_heel"],
  right_ankle: ["right_knee", "right_ankle", "right_heel"],
  spine: ["left_hip", "left_shoulder", "right_hip"],
  head: ["left_shoulder", "nose", "right_shoulder"]
};

const CUE_MAP = {
  left_elbow: "Control left elbow path and full range.",
  right_elbow: "Control right elbow path and full range.",
  left_shoulder: "Keep left shoulder stable.",
  right_shoulder: "Keep right shoulder stable.",
  left_knee: "Track left knee smoothly and reach depth.",
  right_knee: "Track right knee smoothly and reach depth.",
  left_hip: "Sit back and control left hip angle.",
  right_hip: "Sit back and control right hip angle.",
  spine: "Keep trunk stable and neutral.",
  head: "Keep head and neck position steady."
};

const SKELETON_CONNECTIONS = [
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"]
];

let detector;
let rulesById = new Map();

const lastAnalyses = {
  reference: null,
  attempt: null
};

const viewerState = {
  active: null,
  rafId: null
};

const el = {
  branchId: document.getElementById("branchId"),
  exerciseSelect: document.getElementById("exerciseSelect"),
  viewSelect: document.getElementById("viewSelect"),
  refFile: document.getElementById("refFile"),
  attemptFile: document.getElementById("attemptFile"),
  buildTemplateBtn: document.getElementById("buildTemplateBtn"),
  evaluateBtn: document.getElementById("evaluateBtn"),
  templateStatus: document.getElementById("templateStatus"),
  evalStatus: document.getElementById("evalStatus"),
  viewerSource: document.getElementById("viewerSource"),
  loadViewerBtn: document.getElementById("loadViewerBtn"),
  analysisVideo: document.getElementById("analysisVideo"),
  analysisOverlay: document.getElementById("analysisOverlay"),
  viewerStatus: document.getElementById("viewerStatus")
};

const overlayCtx = el.analysisOverlay.getContext("2d");

boot().catch((err) => {
  console.error(err);
  setStatus(el.templateStatus, `Initialization failed: ${err.message}`, "bad");
});

async function boot() {
  setStatus(el.templateStatus, "Loading exercise list...", "warn");
  const response = await fetch("./data/exercise_rules.json");
  const payload = await response.json();

  payload.exercises.forEach((exercise) => {
    rulesById.set(exercise.id, exercise);
    const option = document.createElement("option");
    option.value = exercise.id;
    option.textContent = exercise.name;
    el.exerciseSelect.appendChild(option);
  });

  setStatus(el.templateStatus, "Loading MoveNet model...", "warn");
  await tf.setBackend("webgl");
  await tf.ready();

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
      enableSmoothing: true
    }
  );

  bindEvents();
  syncViewHint();
  refreshViewerSourceOptions();
  setStatus(el.templateStatus, "Ready. Upload trainer reference video.", "ok");
}

function bindEvents() {
  el.exerciseSelect.addEventListener("change", syncViewHint);

  el.buildTemplateBtn.addEventListener("click", async () => {
    try {
      lockButtons(true);
      const ctx = getSessionContext();
      if (!ctx) return;

      const file = el.refFile.files?.[0];
      if (!file) {
        setStatus(el.templateStatus, "Select trainer reference video first.", "bad");
        return;
      }

      setStatus(el.templateStatus, "Analyzing trainer video and generating rules...", "warn");
      const analysis = await analyzeVideo(file, ctx.exercise);

      if (!analysis.quality.passed) {
        setStatus(el.templateStatus, formatQualityFailure(analysis.quality), "bad");
        return;
      }

      if (analysis.reps < (ctx.exercise.minReferenceReps || 1)) {
        setStatus(
          el.templateStatus,
          `Only ${analysis.reps} reps detected. Upload a clip with at least ${ctx.exercise.minReferenceReps || 1} rep(s).`,
          "bad"
        );
        return;
      }

      const template = buildTemplateFromTrainer(analysis, ctx.exercise, ctx.selectedView);
      const saved = saveTemplate(ctx.branchId, ctx.exercise.id, template);

      storeAnalysisResult("reference", file, analysis, ctx);

      setStatus(
        el.templateStatus,
        [
          "Template generated ✅",
          `Branch: ${ctx.branchId}`,
          `Exercise: ${ctx.exercise.name}`,
          `View used: ${ctx.selectedView}`,
          `Version: v${saved.version}`,
          `Detected reps: ${analysis.reps}`
        ].join("\n"),
        "ok"
      );
    } catch (err) {
      console.error(err);
      setStatus(el.templateStatus, `Template build failed: ${err.message}`, "bad");
    } finally {
      lockButtons(false);
    }
  });

  el.evaluateBtn.addEventListener("click", async () => {
    try {
      lockButtons(true);
      const ctx = getSessionContext();
      if (!ctx) return;

      const file = el.attemptFile.files?.[0];
      if (!file) {
        setStatus(el.evalStatus, "Select member attempt video first.", "bad");
        return;
      }

      const template = loadTemplate(ctx.branchId, ctx.exercise.id);
      if (!template) {
        setStatus(el.evalStatus, "No trainer template found. Build reference template first.", "bad");
        return;
      }

      setStatus(el.evalStatus, "Evaluating attempt against trainer-generated template...", "warn");
      const analysis = await analyzeVideo(file, ctx.exercise, template);

      if (!analysis.quality.passed) {
        setStatus(el.evalStatus, formatQualityFailure(analysis.quality), "bad");
        return;
      }

      storeAnalysisResult("attempt", file, analysis, ctx);

      const topFeedback = analysis.feedback.slice(0, 2);
      const feedbackText = topFeedback.length
        ? topFeedback.map((f, i) => `${i + 1}. ${f.cue} (severity ${f.severity.toFixed(2)})`).join("\n")
        : "No major violations detected.";

      setStatus(
        el.evalStatus,
        [
          "Attempt evaluated ✅",
          `Template version: v${template.version}`,
          `Reps counted: ${analysis.reps}`,
          "Feedback:",
          feedbackText
        ].join("\n"),
        analysis.reps > 0 ? "ok" : "warn"
      );
    } catch (err) {
      console.error(err);
      setStatus(el.evalStatus, `Evaluation failed: ${err.message}`, "bad");
    } finally {
      lockButtons(false);
    }
  });

  el.loadViewerBtn.addEventListener("click", () => {
    const value = el.viewerSource.value;
    if (!value) {
      setStatus(el.viewerStatus, "Select an analyzed source first.", "warn");
      return;
    }
    loadViewerByKey(value);
  });

  el.analysisVideo.addEventListener("loadedmetadata", () => {
    resizeOverlayCanvas();
    drawCurrentOverlayFrame();
  });

  el.analysisVideo.addEventListener("timeupdate", drawCurrentOverlayFrame);
  el.analysisVideo.addEventListener("seeked", drawCurrentOverlayFrame);
  el.analysisVideo.addEventListener("pause", () => {
    stopViewerLoop();
    drawCurrentOverlayFrame();
  });
  el.analysisVideo.addEventListener("play", () => {
    startViewerLoop();
  });

  window.addEventListener("resize", () => {
    resizeOverlayCanvas();
    drawCurrentOverlayFrame();
  });
}

function syncViewHint() {
  const exercise = rulesById.get(el.exerciseSelect.value);
  if (!exercise) return;

  if (exercise.allowedViews?.length) {
    el.viewSelect.value = exercise.allowedViews[0];
    el.viewSelect.title = `Preferred views: ${exercise.allowedViews.join(", ")}`;
  }
}

function getSessionContext() {
  const branchId = el.branchId.value.trim().toLowerCase().replace(/\s+/g, "_");
  if (!branchId) {
    setStatus(el.templateStatus, "Branch ID is required.", "bad");
    return null;
  }

  const exercise = rulesById.get(el.exerciseSelect.value);
  if (!exercise) {
    setStatus(el.templateStatus, "Select an exercise.", "bad");
    return null;
  }

  const selectedView = el.viewSelect.value;
  return { branchId, exercise, selectedView };
}

async function analyzeVideo(file, exercise, template = null) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.muted = true;
  video.playsInline = true;

  await waitFor(video, "loadedmetadata");

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const sampleStep = 0.08;
  const angleFrames = [];
  const qualityRaw = {
    totalFrames: 0,
    poseFrames: 0,
    stableAngleFrames: 0
  };

  const importantAngles = exercise.importantAngles || [];
  let startAnchorPrimary = null;

  for (let t = 0; t <= video.duration; t += sampleStep) {
    video.currentTime = Math.min(t, video.duration);
    await waitFor(video, "seeked");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const poses = await detector.estimatePoses(canvas, { maxPoses: 1, flipHorizontal: false });

    qualityRaw.totalFrames += 1;

    const pose = poses?.[0];
    if (!pose?.keypoints?.length) {
      continue;
    }

    qualityRaw.poseFrames += 1;

    const keypointMap = toKeypointMap(pose.keypoints);
    const angles = computeAngles(keypointMap, importantAngles);

    const available = Object.values(angles).filter((v) => typeof v === "number");
    if (!available.length) {
      continue;
    }

    qualityRaw.stableAngleFrames += 1;
    const primaryValue = average(available);

    if (
      startAnchorPrimary === null
      && (
        isFullBodyVisible(keypointMap)
        || hasRequiredAngleVisibility(angles, importantAngles)
      )
    ) {
      startAnchorPrimary = primaryValue;
    }

    angleFrames.push({
      t,
      angles,
      primaryValue,
      keypoints: simplifyKeypoints(pose.keypoints)
    });
  }

  URL.revokeObjectURL(video.src);

  const quality = finalizeQualityLoose(qualityRaw, angleFrames.length);
  if (!quality.passed) {
    return {
      reps: 0,
      quality,
      feedback: [],
      phaseSamples: { start: {}, mid: {}, end: {} },
      overlayFrames: [],
      frameWidth: canvas.width,
      frameHeight: canvas.height,
      duration: video.duration
    };
  }

  const smoothed = movingAverage(angleFrames.map((f) => f.primaryValue), 5);
  const primaryThresholds = template?.primaryThresholds || derivePrimaryThresholds(smoothed, startAnchorPrimary);
  const phaseSamples = collectPhaseSamplesByThresholds(angleFrames, importantAngles, primaryThresholds);

  let reps = 0;
  let feedback = [];
  let overlayTimeline;

  if (template) {
    const evalResult = evaluateWithTemplate(angleFrames, exercise, template);
    reps = evalResult.reps;
    feedback = evalResult.feedback;
    overlayTimeline = evalResult.timeline;
  } else {
    overlayTimeline = buildPrimaryTimeline(angleFrames, primaryThresholds, 3);
    reps = overlayTimeline.reps;
  }

  const overlayFrames = mergeOverlayFrames(angleFrames, overlayTimeline);

  return {
    reps,
    quality,
    feedback,
    phaseSamples,
    angleFrames,
    overlayFrames,
    frameWidth: canvas.width,
    frameHeight: canvas.height,
    duration: video.duration,
    primarySeries: smoothed,
    primaryThresholds,
    startAnchorPrimary
  };
}

function buildTemplateFromTrainer(analysis, exercise, selectedView) {
  const importantAngles = exercise.importantAngles || [];
  const metricRangesByPhase = {};

  const thresholds = analysis.primaryThresholds;

  PHASES.forEach((phase) => {
    metricRangesByPhase[phase] = {};

    importantAngles.forEach((metric) => {
      const arr = analysis.phaseSamples[phase][metric] || [];
      const global = allMetricValues(analysis.phaseSamples, metric);
      const source = arr.length ? arr : global;

      if (!source.length) {
        metricRangesByPhase[phase][metric] = [0, 180];
        return;
      }

      const low = percentile(source, 0.12);
      const high = percentile(source, 0.88);
      const pad = Math.max(5, (high - low) * 0.18);
      metricRangesByPhase[phase][metric] = [clamp(low - pad, 0, 180), clamp(high + pad, 0, 180)];
    });
  });

  return {
    createdAt: new Date().toISOString(),
    viewUsed: selectedView,
    importantAngles,
    primaryThresholds: thresholds,
    metricRangesByPhase
  };
}

function evaluateWithTemplate(angleFrames, exercise, template) {
  const state = {
    currentPhase: "start",
    stableCandidate: null,
    stableCount: 0,
    reps: 0
  };

  const violationStats = {};
  const minDwell = 2;
  const phaseThreshold = 0.35;
  const rangeBufferDeg = 8;

  const phases = [];
  const repCounts = [];
  const repEvents = [];

  angleFrames.forEach((frame) => {
    const phaseScores = scorePhases(
      frame.angles,
      template.metricRangesByPhase,
      template.importantAngles || exercise.importantAngles,
      rangeBufferDeg
    );
    const detectedPhase = pickBestPhase(phaseScores, phaseThreshold);

    let repAdded = false;
    if (detectedPhase) {
      repAdded = advanceStateMachine(state, detectedPhase, minDwell);
      accumulateViolations(
        violationStats,
        frame.angles,
        state.currentPhase,
        template.metricRangesByPhase,
        template.importantAngles || exercise.importantAngles
      );
    } else if (template.primaryThresholds) {
      // Fallback keeps reps moving even when per-angle ranges are briefly noisy.
      const fallbackPhase = classifyPrimaryPhase(frame.primaryValue, template.primaryThresholds);
      repAdded = advanceStateMachine(state, fallbackPhase, minDwell);
    }

    phases.push(detectedPhase || state.currentPhase);
    repCounts.push(state.reps);
    repEvents.push(repAdded);
  });

  return {
    reps: state.reps,
    feedback: buildFeedback(violationStats),
    timeline: { phases, repCounts, repEvents, reps: state.reps }
  };
}

function buildPrimaryTimeline(angleFrames, thresholds, minDwellFrames) {
  const state = {
    currentPhase: "start",
    stableCandidate: null,
    stableCount: 0,
    reps: 0
  };

  const phases = [];
  const repCounts = [];
  const repEvents = [];

  angleFrames.forEach((frame) => {
    const detectedPhase = classifyPrimaryPhase(frame.primaryValue, thresholds);
    const repAdded = advanceStateMachine(state, detectedPhase, Math.max(2, minDwellFrames));
    phases.push(detectedPhase || state.currentPhase);
    repCounts.push(state.reps);
    repEvents.push(repAdded);
  });

  return {
    phases,
    repCounts,
    repEvents,
    reps: state.reps
  };
}

function mergeOverlayFrames(angleFrames, timeline) {
  return angleFrames.map((f, idx) => ({
    t: f.t,
    keypoints: f.keypoints,
    phase: timeline?.phases?.[idx] || "unknown",
    repCount: timeline?.repCounts?.[idx] ?? 0,
    repEvent: timeline?.repEvents?.[idx] || false,
    angles: f.angles
  }));
}

function collectPhaseSamplesByThresholds(angleFrames, importantAngles, providedThresholds = null) {
  const fallback = derivePrimaryThresholds(angleFrames.map((f) => f.primaryValue), null);
  const thresholds = providedThresholds || fallback;
  const store = { start: {}, mid: {}, end: {} };

  angleFrames.forEach((f) => {
    const phase = classifyPrimaryPhase(f.primaryValue, thresholds);
    importantAngles.forEach((metric) => {
      if (typeof f.angles[metric] !== "number") return;
      if (!store[phase][metric]) {
        store[phase][metric] = [];
      }
      store[phase][metric].push(f.angles[metric]);
    });
  });

  return store;
}

function derivePrimaryThresholds(primarySeries, startAnchor = null) {
  if (!primarySeries.length) {
    return {
      low: 60,
      high: 140,
      startAnchor: 140,
      startBand: 10,
      endTarget: 60,
      endBand: 10
    };
  }

  const minV = Math.min(...primarySeries);
  const maxV = Math.max(...primarySeries);
  const p20 = percentile(primarySeries, 0.2);
  const p80 = percentile(primarySeries, 0.8);

  const anchor = typeof startAnchor === "number" ? startAnchor : p80;
  const distanceToMin = Math.abs(anchor - minV);
  const distanceToMax = Math.abs(anchor - maxV);
  const endTarget = distanceToMin >= distanceToMax ? minV : maxV;

  const span = Math.max(8, Math.abs(anchor - endTarget));
  const startBand = clamp(span * 0.24, 8, 28);
  const endBand = clamp(span * 0.24, 8, 28);

  const low = Math.min(p20, p80);
  const high = Math.max(p20, p80);

  return {
    low,
    high,
    startAnchor: clamp(anchor, 0, 180),
    startBand,
    endTarget: clamp(endTarget, 0, 180),
    endBand
  };
}

function classifyPrimaryPhase(value, thresholds) {
  const {
    startAnchor = thresholds.high,
    startBand = 10,
    endTarget = thresholds.low,
    endBand = 10,
    low = thresholds.low,
    high = thresholds.high
  } = thresholds;

  const nearStart = Math.abs(value - startAnchor) <= startBand;
  const nearEnd = Math.abs(value - endTarget) <= endBand;

  if (nearStart && !nearEnd) return "start";
  if (nearEnd && !nearStart) return "end";

  if (value >= high) return "start";
  if (value <= low) return "end";
  return "mid";
}

function scorePhases(angles, metricRangesByPhase, importantAngles, rangeBufferDeg = 0) {
  const phaseScores = { start: 0, mid: 0, end: 0 };

  PHASES.forEach((phase) => {
    let pass = 0;
    let total = 0;

    (importantAngles || []).forEach((metric) => {
      const value = angles[metric];
      const range = metricRangesByPhase?.[phase]?.[metric];
      if (typeof value !== "number" || !range) return;

      total += 1;
      const minR = range[0] - rangeBufferDeg;
      const maxR = range[1] + rangeBufferDeg;
      if (value >= minR && value <= maxR) {
        pass += 1;
      }
    });

    phaseScores[phase] = total ? pass / total : 0;
  });

  return phaseScores;
}

function pickBestPhase(phaseScores, threshold) {
  let bestPhase = null;
  let bestScore = 0;

  PHASES.forEach((phase) => {
    if (phaseScores[phase] > bestScore) {
      bestScore = phaseScores[phase];
      bestPhase = phase;
    }
  });

  return bestScore >= threshold ? bestPhase : null;
}

function advanceStateMachine(state, detectedPhase, minDwellFrames) {
  let repAdded = false;

  if (state.stableCandidate !== detectedPhase) {
    state.stableCandidate = detectedPhase;
    state.stableCount = 1;
    return false;
  }

  state.stableCount += 1;
  if (state.stableCount < minDwellFrames) return false;
  if (state.currentPhase === detectedPhase) return false;
  if (!isValidTransition(state.currentPhase, detectedPhase)) return false;

  if (state.currentPhase === "end" && detectedPhase === "start") {
    state.reps += 1;
    repAdded = true;
  }

  state.currentPhase = detectedPhase;
  return repAdded;
}

function isValidTransition(from, to) {
  return (
    (from === "start" && to === "mid") ||
    (from === "mid" && to === "end") ||
    (from === "end" && to === "start")
  );
}

function accumulateViolations(violationStats, angles, phase, metricRangesByPhase, importantAngles) {
  (importantAngles || []).forEach((metric) => {
    const value = angles[metric];
    const range = metricRangesByPhase?.[phase]?.[metric];

    if (typeof value !== "number" || !range) {
      return;
    }

    if (!violationStats[metric]) {
      violationStats[metric] = { severity: 0, streak: 0, maxStreak: 0 };
    }

    const distance = outsideDistance(value, range[0], range[1]);
    if (distance <= 0) {
      violationStats[metric].streak = 0;
      return;
    }

    violationStats[metric].streak += 1;
    violationStats[metric].maxStreak = Math.max(violationStats[metric].maxStreak, violationStats[metric].streak);
    violationStats[metric].severity += distance / 10;
  });
}

function buildFeedback(violationStats) {
  return Object.entries(violationStats)
    .filter(([, stat]) => stat.maxStreak >= 5)
    .map(([metric, stat]) => ({
      metric,
      cue: CUE_MAP[metric] || `Improve ${metric.replace("_", " ")} control.`,
      severity: stat.severity
    }))
    .sort((a, b) => b.severity - a.severity);
}

function computeAngles(keypointMap, metricNames) {
  const out = {};

  (metricNames || []).forEach((metric) => {
    const triplet = JOINT_MAP[metric];
    if (!triplet) return;

    const [aName, bName, cName] = triplet;
    const a = keypointMap[aName];
    const b = keypointMap[bName];
    const c = keypointMap[cName];

    if (!a || !b || !c || a.score < 0.2 || b.score < 0.2 || c.score < 0.2) {
      return;
    }

    out[metric] = calculateAngle(a, b, c);
  });

  return out;
}

function calculateAngle(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;

  const denom = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
  if (denom < 1e-6) return 0;

  const cosine = clamp((abx * cbx + aby * cby) / denom, -1, 1);
  return (Math.acos(cosine) * 180) / Math.PI;
}

function toKeypointMap(keypoints) {
  const map = {};
  keypoints.forEach((kp) => {
    map[kp.name] = { x: kp.x, y: kp.y, score: kp.score ?? 0 };
  });
  return map;
}

function simplifyKeypoints(keypoints) {
  return keypoints.map((kp) => ({
    name: kp.name,
    x: kp.x,
    y: kp.y,
    score: kp.score ?? 0
  }));
}

function keypointArrayToMap(keypoints) {
  const map = {};
  keypoints.forEach((kp) => {
    map[kp.name] = kp;
  });
  return map;
}

function isFullBodyVisible(keypointMap) {
  const required = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_ankle", "right_ankle"
  ];

  const visible = required.filter((name) => (keypointMap[name]?.score || 0) >= 0.25).length;
  return visible >= 6;
}

function hasRequiredAngleVisibility(angles, importantAngles) {
  if (!importantAngles?.length) return false;
  const available = importantAngles.filter((metric) => typeof angles[metric] === "number").length;
  const needed = Math.max(1, Math.ceil(importantAngles.length * 0.6));
  return available >= needed;
}

function finalizeQualityLoose(qualityRaw, validAngleFrames) {
  const total = Math.max(qualityRaw.totalFrames, 1);
  const poseRatio = qualityRaw.poseFrames / total;
  const angleRatio = validAngleFrames / total;

  const reasons = [];
  if (poseRatio < 0.35) {
    reasons.push("Pose detection too unstable. Improve lighting and keep full body visible.");
  }
  if (angleRatio < 0.25) {
    reasons.push("Important angles were not trackable enough. Try a cleaner camera angle.");
  }

  return {
    passed: reasons.length === 0,
    ratios: { poseRatio, angleRatio },
    reasons
  };
}

function formatQualityFailure(quality) {
  return [
    "Upload quality failed ❌",
    ...quality.reasons,
    `Pose frames: ${(quality.ratios.poseRatio * 100).toFixed(0)}%`,
    `Trackable angle frames: ${(quality.ratios.angleRatio * 100).toFixed(0)}%`
  ].join("\n");
}

function saveTemplate(branchId, exerciseId, template) {
  const key = storageKey(branchId, exerciseId);
  const previous = JSON.parse(localStorage.getItem(key) || "null");
  const version = (previous?.version || 0) + 1;

  const payload = {
    ...template,
    version
  };

  localStorage.setItem(key, JSON.stringify(payload));
  return payload;
}

function loadTemplate(branchId, exerciseId) {
  const raw = localStorage.getItem(storageKey(branchId, exerciseId));
  return raw ? JSON.parse(raw) : null;
}

function storageKey(branchId, exerciseId) {
  return `branch_template::${branchId}::${exerciseId}`;
}

function waitFor(target, eventName) {
  return new Promise((resolve, reject) => {
    const onDone = () => {
      target.removeEventListener(eventName, onDone);
      target.removeEventListener("error", onErr);
      resolve();
    };
    const onErr = () => {
      target.removeEventListener(eventName, onDone);
      target.removeEventListener("error", onErr);
      reject(new Error(`Failed while waiting for ${eventName}`));
    };

    target.addEventListener(eventName, onDone, { once: true });
    target.addEventListener("error", onErr, { once: true });
  });
}

function average(arr) {
  if (!arr.length) return 0;
  return arr.reduce((sum, v) => sum + v, 0) / arr.length;
}

function movingAverage(arr, windowSize) {
  if (!arr.length) return [];
  const out = [];
  const half = Math.floor(windowSize / 2);

  for (let i = 0; i < arr.length; i += 1) {
    const start = Math.max(0, i - half);
    const end = Math.min(arr.length - 1, i + half);
    let sum = 0;
    let count = 0;
    for (let j = start; j <= end; j += 1) {
      sum += arr[j];
      count += 1;
    }
    out.push(sum / Math.max(count, 1));
  }
  return out;
}

function percentile(arr, p) {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor((sorted.length - 1) * p);
  return sorted[Math.max(0, Math.min(idx, sorted.length - 1))];
}

function allMetricValues(phaseSamples, metric) {
  const values = [];
  PHASES.forEach((phase) => {
    const arr = phaseSamples?.[phase]?.[metric] || [];
    values.push(...arr);
  });
  return values;
}

function outsideDistance(value, min, max) {
  if (value < min) return min - value;
  if (value > max) return value - max;
  return 0;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lockButtons(locked) {
  el.buildTemplateBtn.disabled = locked;
  el.evaluateBtn.disabled = locked;
}

function setStatus(node, text, level = "warn") {
  node.textContent = text;
  node.classList.remove("ok", "warn", "bad");
  node.classList.add(level);
}

function storeAnalysisResult(kind, file, analysis, ctx) {
  if (lastAnalyses[kind]?.videoUrl) {
    URL.revokeObjectURL(lastAnalyses[kind].videoUrl);
  }

  const videoUrl = URL.createObjectURL(file);
  const label = `${kind === "reference" ? "Trainer" : "Attempt"} | ${ctx.exercise.name} | ${ctx.branchId}`;

  lastAnalyses[kind] = {
    key: kind,
    label,
    videoUrl,
    frameWidth: analysis.frameWidth,
    frameHeight: analysis.frameHeight,
    duration: analysis.duration,
    overlayFrames: analysis.overlayFrames,
    exerciseName: ctx.exercise.name,
    branchId: ctx.branchId
  };

  refreshViewerSourceOptions();
  loadViewerByKey(kind);
}

function refreshViewerSourceOptions() {
  const current = el.viewerSource.value;
  const options = [{ value: "", label: "Select analyzed video..." }];

  if (lastAnalyses.reference) {
    options.push({ value: "reference", label: `Trainer Reference - ${lastAnalyses.reference.exerciseName}` });
  }
  if (lastAnalyses.attempt) {
    options.push({ value: "attempt", label: `Member Attempt - ${lastAnalyses.attempt.exerciseName}` });
  }

  el.viewerSource.innerHTML = "";
  options.forEach((opt) => {
    const node = document.createElement("option");
    node.value = opt.value;
    node.textContent = opt.label;
    el.viewerSource.appendChild(node);
  });

  if (options.some((o) => o.value === current)) {
    el.viewerSource.value = current;
  } else if (options.length > 1) {
    el.viewerSource.value = options[1].value;
  }
}

function loadViewerByKey(key) {
  const data = lastAnalyses[key];
  if (!data) {
    setStatus(el.viewerStatus, "No analysis data available for selected source.", "warn");
    return;
  }

  viewerState.active = data;
  el.analysisVideo.src = data.videoUrl;
  el.analysisVideo.currentTime = 0;
  resizeOverlayCanvas();

  setStatus(
    el.viewerStatus,
    [
      `Loaded: ${data.label}`,
      "Overlay legend:",
      "- Cyan lines: skeleton",
      "- Green joints: detected keypoints",
      "- HUD: current phase + rep count",
      "- REP +1 flashes exactly when FSM counts a rep"
    ].join("\n"),
    "ok"
  );
}

function startViewerLoop() {
  stopViewerLoop();
  const tick = () => {
    drawCurrentOverlayFrame();
    if (!el.analysisVideo.paused && !el.analysisVideo.ended) {
      viewerState.rafId = requestAnimationFrame(tick);
    }
  };
  viewerState.rafId = requestAnimationFrame(tick);
}

function stopViewerLoop() {
  if (viewerState.rafId) {
    cancelAnimationFrame(viewerState.rafId);
    viewerState.rafId = null;
  }
}

function resizeOverlayCanvas() {
  const rect = el.analysisVideo.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  el.analysisOverlay.width = Math.round(rect.width);
  el.analysisOverlay.height = Math.round(rect.height);
}

function drawCurrentOverlayFrame() {
  const active = viewerState.active;
  if (!active || !active.overlayFrames?.length) {
    clearOverlay();
    return;
  }

  const t = el.analysisVideo.currentTime || 0;
  const frame = findNearestFrame(active.overlayFrames, t);
  if (!frame) {
    clearOverlay();
    return;
  }

  const cw = el.analysisOverlay.width;
  const ch = el.analysisOverlay.height;
  clearOverlay();

  const sx = cw / Math.max(1, active.frameWidth);
  const sy = ch / Math.max(1, active.frameHeight);
  const kmap = keypointArrayToMap(frame.keypoints || []);

  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = "rgba(80, 225, 255, 0.85)";

  SKELETON_CONNECTIONS.forEach(([aName, bName]) => {
    const a = kmap[aName];
    const b = kmap[bName];
    if (!a || !b || a.score < 0.25 || b.score < 0.25) return;

    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x * sx, a.y * sy);
    overlayCtx.lineTo(b.x * sx, b.y * sy);
    overlayCtx.stroke();
  });

  overlayCtx.fillStyle = "rgba(80, 255, 120, 0.95)";
  Object.values(kmap).forEach((k) => {
    if (!k || k.score < 0.25) return;
    overlayCtx.beginPath();
    overlayCtx.arc(k.x * sx, k.y * sy, 4, 0, Math.PI * 2);
    overlayCtx.fill();
  });

  drawHud(frame, t, cw, ch);
}

function drawHud(frame, t, cw, ch) {
  const phase = (frame.phase || "unknown").toUpperCase();
  const repText = `REPS: ${frame.repCount}`;
  const timeText = `T: ${t.toFixed(2)}s`;

  overlayCtx.fillStyle = "rgba(0, 0, 0, 0.55)";
  overlayCtx.fillRect(12, 12, 220, 92);

  overlayCtx.fillStyle = "#e8eef5";
  overlayCtx.font = "bold 18px Segoe UI";
  overlayCtx.fillText(repText, 24, 40);

  overlayCtx.font = "14px Segoe UI";
  overlayCtx.fillText(`PHASE: ${phase}`, 24, 62);
  overlayCtx.fillText(timeText, 24, 82);

  if (frame.repEvent) {
    overlayCtx.fillStyle = "rgba(56, 193, 114, 0.9)";
    overlayCtx.fillRect(cw - 180, 18, 160, 42);
    overlayCtx.fillStyle = "#0f1115";
    overlayCtx.font = "bold 22px Segoe UI";
    overlayCtx.fillText("REP +1", cw - 160, 47);
  }
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, el.analysisOverlay.width, el.analysisOverlay.height);
}

function findNearestFrame(frames, timeSec) {
  if (!frames.length) return null;

  let low = 0;
  let high = frames.length - 1;

  while (low <= high) {
    const mid = (low + high) >> 1;
    const t = frames[mid].t;
    if (t === timeSec) {
      return frames[mid];
    }
    if (t < timeSec) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  if (low <= 0) return frames[0];
  if (low >= frames.length) return frames[frames.length - 1];

  const prev = frames[low - 1];
  const next = frames[low];
  return Math.abs(prev.t - timeSec) <= Math.abs(next.t - timeSec) ? prev : next;
}
