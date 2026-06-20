"use strict";

const MODEL_YOLO_URL = "/models/yolo26s.onnx";
const MODEL_FACE_URL = "/models/face_landmarker.task";

const VIDEO_TARGET_WIDTH = 640; 
const VIDEO_TARGET_HEIGHT = 480;
const YOLO_SIZE = 640;
const VIOLATION_COOLDOWN_MS = 3000;
const MAX_INCIDENT_CARDS = 100;

const YAW_THRESHOLD_DEGREES = 28;

function computeYawFromMatrix(matrixObject) {
    if (!matrixObject || !matrixObject.data) return 0;
    const matrix = matrixObject.data;
    if (!matrix || matrix.length < 16) return 0;
    const m00 = Number(matrix[0]);
    const m10 = Number(matrix[4]);
    if (!Number.isFinite(m00) || !Number.isFinite(m10)) return 0;
    return Math.atan2(m10, m00) * (180 / Math.PI);
}

const video = document.getElementById("video");
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d", { alpha: true });

const yoloCanvas = document.getElementById("yoloCanvas"); 
const yoloCtx = yoloCanvas.getContext("2d", { willReadFrequently: true });

const rawCanvas = document.getElementById("rawCanvas");
const rawCtx = rawCanvas.getContext("2d", { willReadFrequently: true });

const snapshotCanvas = document.getElementById("snapshotCanvas");
const snapshotCtx = snapshotCanvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const purgeBtn = document.getElementById("purgeBtn");

const fallback = document.getElementById("fallback");
const coreStatus = document.getElementById("coreStatus");
const yoloState = document.getElementById("yoloState");
const faceState = document.getElementById("faceState");
const yawReadout = document.getElementById("yawReadout");
const incidentCount = document.getElementById("incidentCount");
const logFeed = document.getElementById("logFeed");
const emptyFeed = document.getElementById("emptyFeed");
const feedStatus = document.getElementById("feedStatus");
const sessionMeta = document.getElementById("sessionMeta");
const isolationBadge = document.getElementById("isolationBadge");
const exportZipBtn = document.getElementById("exportZipBtn");
const infoBtn = document.getElementById("infoBtn");
const infoModal = document.getElementById("infoModal");
const infoCloseBtn = document.getElementById("infoCloseBtn");
const gitRepoLink = document.getElementById("gitRepoLink");
const themeToggleBtn = document.getElementById("themeToggleBtn");

let yoloWorker = null;
let faceLandmarker = null;

let yoloReady = false;
let faceReady = false;
let yoloBusy = false;
let faceBusy = false;

let sessionActive = false;
let animationFrameId = null;
let mediaStream = null;

let currentBoundingBoxes = [];
let currentFaceLandmarks = [];
let currentYaw = 0;

let lastViolationTimes = {};
let incidentTotal = 0;

const forbiddenLabels = {
    63: "Laptop / Secondary Screen",
    67: "Cell Phone"
};

document.addEventListener("DOMContentLoaded", bootSystem);
startBtn.addEventListener("click", startLiveSession);
stopBtn.addEventListener("click", terminateSession);
purgeBtn.addEventListener("click", purgeLogs);
exportZipBtn.addEventListener("click", exportViolationsAsZip);
if (infoBtn) infoBtn.addEventListener("click", openInfoModal);
if (infoCloseBtn) infoCloseBtn.addEventListener("click", closeInfoModal);
if (infoModal) infoModal.addEventListener("click", (event) => {
    if (event.target === infoModal) {
        closeInfoModal();
    }
});
if (themeToggleBtn) themeToggleBtn.addEventListener("click", toggleTheme);

function bootSystem() {
    updateIsolationBadge();
    initializeWorkers();
    initializeFaceSystem();
    initializeInfoPanel();
}

function initializeInfoPanel() {
    if (gitRepoLink) {
        gitRepoLink.href = "https://github.com/your-repo-url";
        gitRepoLink.textContent = "Edit project repo link";
    }
    applySavedTheme();
}

function openInfoModal() {
    if (!infoModal) return;
    infoModal.classList.remove("hidden");
    document.body.style.overflow = "hidden";
}

function closeInfoModal() {
    if (!infoModal) return;
    infoModal.classList.add("hidden");
    document.body.style.overflow = "";
}

function toggleTheme() {
    const isDark = document.body.classList.toggle("dark-mode");
    if (themeToggleBtn) {
        themeToggleBtn.textContent = isDark ? "Light Mode" : "Dark Mode";
    }
    localStorage.setItem("dashboardTheme", isDark ? "dark" : "light");
}

function applySavedTheme() {
    const savedTheme = localStorage.getItem("dashboardTheme");
    const shouldUseDark = savedTheme === "dark" || (!savedTheme && window.matchMedia?.("(prefers-color-scheme: dark)").matches);
    if (shouldUseDark) {
        document.body.classList.add("dark-mode");
        if (themeToggleBtn) themeToggleBtn.textContent = "Light Mode";
    } else if (themeToggleBtn) {
        themeToggleBtn.textContent = "Dark Mode";
    }
}

function updateIsolationBadge() {
    if (self.crossOriginIsolated) {
        isolationBadge.textContent = "Isolation: true";
    } else {
        isolationBadge.textContent = "Isolation: false";
        isolationBadge.style.color = "#fca5a5";
        isolationBadge.style.borderColor = "rgba(248,113,113,0.35)";
        isolationBadge.style.background = "rgba(239,68,68,0.12)";
    }
}

function initializeWorkers() {
    const epoch = Date.now();

    yoloWorker = new Worker(`/workers/yoloWorker.js?t=${epoch}`);

    yoloWorker.onmessage = handleYoloMessage;
    yoloWorker.onerror = handleYoloError;

    yoloWorker.postMessage({
        action: "INIT",
        modelUrl: MODEL_YOLO_URL
    });

    if (coreStatus) coreStatus.textContent = "Edge Core Initializing";
}

async function initializeFaceSystem() {
    if (faceState) faceState.textContent = "LOADING";
    try {
        const vision = await import(`/vendor/mediapipe/vision_bundle.mjs?t=${Date.now()}`);
        const { FaceLandmarker, FilesetResolver } = vision;

        if (!FaceLandmarker || !FilesetResolver) {
            throw new Error("MediaPipe bundle missing FaceLandmarker or FilesetResolver");
        }

        const visionFileset = await FilesetResolver.forVisionTasks("/vendor/mediapipe/wasm");

        faceLandmarker = await FaceLandmarker.createFromOptions(visionFileset, {
            baseOptions: {
                modelAssetPath: MODEL_FACE_URL,
                delegate: "CPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true
        });

        faceReady = true;
        faceState.textContent = "READY";
        unlockStartIfReady();
    } catch (err) {
        setFaceSystemError(err);
    }
}

function setFaceSystemError(err) {
    const message = err && err.message ? err.message : String(err);
    faceReady = false;
    faceBusy = false;
    if (faceState) faceState.textContent = "ERROR";
    if (faceState) {
        faceState.style.color = "#fca5a5";
        faceState.style.borderColor = "rgba(248,113,113,0.35)";
        faceState.style.background = "rgba(239,68,68,0.12)";
    }
    console.error("Face system error:", message);
    registerViolation("FACE_SYSTEM_ERROR", "Face system initialization failed", message);
    setTimeout(() => {
        console.info("Retrying face system initialization...");
        initializeFaceSystem();
    }, 2000);
}

function handleYoloMessage(event) {
    const message = event.data || {};

    if (message.status === "READY_WEBGPU") {
        yoloReady = true;
        if (yoloState) yoloState.textContent = "READY_WEBGPU";
        unlockStartIfReady();
        return;
    }

    if (message.status === "READY_WASM") {
        yoloReady = true;
        if (yoloState) yoloState.textContent = "READY_WASM";
        unlockStartIfReady();
        return;
    }

    if (message.status === "RESULT") {
        yoloBusy = false;
        if (yoloState) yoloState.textContent = yoloReady ? "READY" : "INIT";

        currentBoundingBoxes = normalizeYoloResult(message.data || []);

        for (const box of currentBoundingBoxes) {
            if (box.violation) {
                registerViolation(
                    `OBJECT_${box.classId}`,
                    `${box.label} Detected`,
                    `Confidence ${(box.confidence * 100).toFixed(1)}%`
                );
            }
        }

        return;
    }

    if (message.status === "ERROR") {
        yoloBusy = false;
        if (yoloState) yoloState.textContent = "ERROR";
        console.error("YOLO worker error:", message.error);
        try {
            if (yoloWorker) {
                yoloWorker.terminate();
            }
        } catch (e) {}

        setTimeout(() => {
            console.info("Reinitializing YOLO worker after error...");
            initializeWorkers();
        }, 1000);
    }
}

function handleFaceMessage(event) {
    const message = event.data || {};

    if (message.status === "READY") {
        faceReady = true;
        if (faceState) faceState.textContent = "READY";
        unlockStartIfReady();
        return;
    }

    if (message.status === "RESULT") {
        faceBusy = false;
        if (faceState) faceState.textContent = "READY";

        const data = message.data || {};
        currentFaceLandmarks = data.faceLandmarks || [];
        currentYaw = Number(data.yaw || 0);
        if (yawReadout) yawReadout.textContent = `${currentYaw.toFixed(2)}°`;

        if (data.violationType === "FACE_ABSENT") {
            registerViolation(
                "FACE_ABSENT",
                "Candidate Absent From Frame",
                "No valid face detected"
            );
        }

        if (data.violationType === "FACE_AWAY") {
            registerViolation(
                "FACE_AWAY",
                "Face Turned Away From Screen",
                `Yaw ${currentYaw.toFixed(2)}°`
            );
        }

        return;
    }

    if (message.status === "ERROR") {
        faceBusy = false;
        if (faceState) faceState.textContent = "ERROR";
        console.error("Face worker error:", message.error);
    }
}

function handleYoloError(error) {
    yoloBusy = false;
    if (yoloState) yoloState.textContent = "ERROR";
    console.error("YOLO worker crashed:", error);
    try {
        if (yoloWorker) {
            yoloWorker.terminate();
        }
    } catch (e) {}

    setTimeout(() => {
        console.info("Reinitializing YOLO worker after crash...");
        initializeWorkers();
    }, 1000);
}

function handleFaceError(error) {
    faceBusy = false;
    if (faceState) faceState.textContent = "ERROR";
    console.error("Face worker crashed:", error);
}

function unlockStartIfReady() {
    if (yoloReady && faceReady) {
        if (startBtn) startBtn.disabled = false;
        if (coreStatus) coreStatus.textContent = "Edge Core Active";
    }
}

async function startLiveSession() {
    if (sessionActive) return;

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                width: { ideal: VIDEO_TARGET_WIDTH },
                height: { ideal: VIDEO_TARGET_HEIGHT },
                frameRate: { ideal: 30, max: 60 },
                facingMode: "user"
            }
        });

        video.srcObject = mediaStream;

        await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
        });

        await video.play();

        sessionActive = true;
        fallback.classList.add("hidden");
        startBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");

        resizeCanvases();

        sessionMeta.textContent = `${video.videoWidth}×${video.videoHeight} active camera stream`;

        animationFrameId = requestAnimationFrame(processFrameLoop);
    } catch (error) {
        console.error("Camera initialization failed:", error);

        registerViolation(
            "CAMERA_ERROR",
            "Camera Initialization Failed",
            error.message || "Unable to access camera"
        );
    }
}

function terminateSession() {
    sessionActive = false;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (mediaStream) {
        for (const track of mediaStream.getTracks()) {
            track.stop();
        }

        mediaStream = null;
    }

    video.srcObject = null;

    fallback.classList.remove("hidden");
    startBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");

    currentBoundingBoxes = [];
    currentFaceLandmarks = [];
    currentYaw = 0;

    yawReadout.textContent = "0.00°";
    sessionMeta.textContent = "Session Pipeline Disengaged";

    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function resizeCanvases() {
    const width = video.videoWidth || VIDEO_TARGET_WIDTH;
    const height = video.videoHeight || VIDEO_TARGET_HEIGHT;

    overlayCanvas.width = width;
    overlayCanvas.height = height;

    rawCanvas.width = width;
    rawCanvas.height = height;

    snapshotCanvas.width = width;
    snapshotCanvas.height = height;

    yoloCanvas.width = YOLO_SIZE;
    yoloCanvas.height = YOLO_SIZE;
}

function processFrameLoop(timestamp) {
    if (!sessionActive) return;

    drawOverlay();

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (yoloReady && !yoloBusy) {
            dispatchYoloInference();
        }

        if (faceReady && !faceBusy) {
            dispatchFaceInference(timestamp);
        }
    }

    animationFrameId = requestAnimationFrame(processFrameLoop);
}

function dispatchYoloInference() {
    try {
        yoloBusy = true;
        if (yoloState) yoloState.textContent = "INFERENCE";

        yoloCtx.drawImage(video, 0, 0, YOLO_SIZE, YOLO_SIZE);

        const imageData = yoloCtx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE);
        const tensorData = imageDataToCHWTensor(imageData);

        yoloWorker.postMessage(
            {
                action: "INFERENCE",
                tensorData,
                dims: [1, 3, YOLO_SIZE, YOLO_SIZE]
            },
            [tensorData.buffer]
        );
    } catch (error) {
        yoloBusy = false;
        console.error("YOLO dispatch failed:", error);
    }
}

function dispatchFaceInference(timestamp) {
    if (!faceLandmarker) {
        faceBusy = false;
        return;
    }

    (async () => {
        try {
            faceBusy = true;
            if (faceState) faceState.textContent = "PROCESS";

            rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);

            const safeTimestamp = Math.round(timestamp || performance.now());

            const result = await faceLandmarker.detectForVideo(rawCanvas, safeTimestamp);

            faceBusy = false;
            if (faceState) faceState.textContent = "READY";

            const faceLandmarks = result.faceLandmarks || [];
            const facialTransformationMatrixes = result.facialTransformationMatrixes || [];

            currentFaceLandmarks = faceLandmarks;

            let yaw = 0;
            let violationType = null;

            if (!faceLandmarks.length) {
                violationType = "FACE_ABSENT";
            } else if (facialTransformationMatrixes.length) {
                yaw = computeYawFromMatrix(facialTransformationMatrixes[0]);
                if (Math.abs(yaw) > YAW_THRESHOLD_DEGREES) {
                    violationType = "FACE_AWAY";
                }
            }

            currentYaw = Number(yaw || 0);
            if (yawReadout) yawReadout.textContent = `${currentYaw.toFixed(2)}°`;

            if (violationType === "FACE_ABSENT") {
                registerViolation(
                    "FACE_ABSENT",
                    "Candidate Absent From Frame",
                    "No valid face detected"
                );
            }

            if (violationType === "FACE_AWAY") {
                registerViolation(
                    "FACE_AWAY",
                    "Face Turned Away From Screen",
                    `Yaw ${currentYaw.toFixed(2)}°`
                );
            }
        } catch (error) {
            faceBusy = false;
            setFaceSystemError(error);
        }
    })();
}

function imageDataToCHWTensor(imageData) {
    const { data, width, height } = imageData;
    const planeSize = width * height;
    const tensor = new Float32Array(3 * planeSize);

    for (let i = 0, pixel = 0; i < data.length; i += 4, pixel++) {
        tensor[pixel] = data[i] / 255.0;
        tensor[planeSize + pixel] = data[i + 1] / 255.0;
        tensor[planeSize * 2 + pixel] = data[i + 2] / 255.0;
    }

    return tensor;
}

function normalizeYoloResult(flatData) {
    const boxes = [];

    for (let i = 0; i + 5 < flatData.length; i += 6) {
        const x1 = Number(flatData[i]);
        const y1 = Number(flatData[i + 1]);
        const x2 = Number(flatData[i + 2]);
        const y2 = Number(flatData[i + 3]);
        const confidence = Number(flatData[i + 4]);
        const classId = Number(flatData[i + 5]);

        if (!Number.isFinite(confidence) || confidence < 0.25) {
            continue;
        }

        const label = forbiddenLabels[classId] || "Detected Object";
        const violation = forbiddenLabels[classId] !== undefined;

        boxes.push({
            x1,
            y1,
            x2,
            y2,
            confidence,
            classId,
            label,
            violation
        });
    }

    return boxes;
}

function drawOverlay() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    drawBoundingBoxes();
    drawFaceLandmarks();
    drawTelemetryStrip();
}

function drawBoundingBoxes() {
    if (!currentBoundingBoxes.length) return;

    const scaleX = overlayCanvas.width / YOLO_SIZE;
    const scaleY = overlayCanvas.height / YOLO_SIZE;

    for (const box of currentBoundingBoxes) {
        if (!box.violation) continue;

        const x = box.x1 * scaleX;
        const y = box.y1 * scaleY;
        const width = (box.x2 - box.x1) * scaleX;
        const height = (box.y2 - box.y1) * scaleY;

        overlayCtx.save();
        overlayCtx.lineWidth = 2;
        overlayCtx.strokeStyle = "#22d3ee";
        overlayCtx.fillStyle = "rgba(34, 211, 238, 0.16)";

        overlayCtx.strokeRect(x, y, width, height);
        overlayCtx.fillRect(x, y, width, height);

        const label = `${box.label} ${(box.confidence * 100).toFixed(1)}%`;

        overlayCtx.font = "bold 13px system-ui, sans-serif";
        const labelWidth = Math.min(overlayCanvas.width - 8, overlayCtx.measureText(label).width + 18);
        const labelY = Math.max(4, y - 30);

        overlayCtx.fillStyle = "rgba(56, 189, 248, 0.92)";
        overlayCtx.fillRect(x, labelY, labelWidth, 24);

        overlayCtx.fillStyle = "#0f172a";
        overlayCtx.fillText(label, x + 9, labelY + 16);

        overlayCtx.restore();
    }
}

function drawFaceLandmarks() {
    if (!currentFaceLandmarks.length) return;

    const landmarks = currentFaceLandmarks[0] || [];
    if (!landmarks.length) return;

    overlayCtx.save();

    for (let i = 0; i < landmarks.length; i += 3) {
        const point = landmarks[i];

        if (!point) continue;

        const x = point.x * overlayCanvas.width;
        const y = point.y * overlayCanvas.height;

        overlayCtx.beginPath();
        overlayCtx.arc(x, y, 1.35, 0, Math.PI * 2);
        overlayCtx.fillStyle = "#38bdf8";
        overlayCtx.fill();
    }

    const importantIndexes = [33, 133, 263, 362, 1, 4, 152];

    overlayCtx.fillStyle = "#0ea5e9";

    for (const index of importantIndexes) {
        const point = landmarks[index];

        if (!point) continue;

        overlayCtx.beginPath();
        overlayCtx.arc(
            point.x * overlayCanvas.width,
            point.y * overlayCanvas.height,
            3,
            0,
            Math.PI * 2
        );
        overlayCtx.fill();
    }

    overlayCtx.restore();
}

function drawTelemetryStrip() {
    if (!overlayCanvas.width || !overlayCanvas.height) return;

    overlayCtx.save();

    const stripHeight = 36;

    overlayCtx.fillStyle = "rgba(2, 6, 23, 0.72)";
    overlayCtx.fillRect(
        0,
        overlayCanvas.height - stripHeight,
        overlayCanvas.width,
        stripHeight
    );

    overlayCtx.font = "bold 13px system-ui, sans-serif";
    overlayCtx.fillStyle = Math.abs(currentYaw) > 28 ? "#f87171" : "#67e8f9";

    const text = `Yaw: ${currentYaw.toFixed(2)}° | YOLO: ${
        yoloBusy ? "INFERENCE" : "READY"
    } | FACE: ${faceBusy ? "PROCESS" : "READY"}`;

    overlayCtx.fillText(text, 14, overlayCanvas.height - 13);

    overlayCtx.restore();
}

function registerViolation(type, title, detail) {
    const now = Date.now();
    const last = lastViolationTimes[type] || 0;

    if (now - last < VIOLATION_COOLDOWN_MS) {
        return;
    }

    lastViolationTimes[type] = now;
    incidentTotal += 1;
    if (incidentCount) incidentCount.textContent = String(incidentTotal);

    if (feedStatus) {
        feedStatus.textContent = "⚠️ Active Compliance Events";
        feedStatus.style.color = "#fca5a5";
        feedStatus.style.borderColor = "rgba(248,113,113,0.35)";
        feedStatus.style.background = "rgba(239,68,68,0.12)";
    }

    if (emptyFeed) {
        emptyFeed.classList.add("hidden");
    }

    const snapshot = createSnapshot();

    const card = document.createElement("article");
    card.className = "incident-card";

    const timeString = new Date(now).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });

    card.innerHTML = `
        <div class="incident-inner">
            <img src="${snapshot}" alt="Violation snapshot" />
            <div>
                <h3 class="incident-title">${escapeHtml(title)}</h3>
                <p class="incident-detail">${escapeHtml(detail || "")}</p>
                <div class="incident-tag">${escapeHtml(type)}</div>
                <p class="incident-time">${timeString}</p>
                <div class="incident-actions">
                    <a href="${snapshot}" download="${timeString}-${type}.jpg">Download Image</a>
                </div>
            </div>
        </div>
    `;
    card.dataset.snapshot = snapshot;
    card.dataset.violationType = type;

    logFeed.prepend(card);

    const cards = logFeed.querySelectorAll(".incident-card");

    if (cards.length > MAX_INCIDENT_CARDS) {
        cards[cards.length - 1].remove();
    }
}

function createSnapshot() {
    const width = video.videoWidth || rawCanvas.width || VIDEO_TARGET_WIDTH;
    const height = video.videoHeight || rawCanvas.height || VIDEO_TARGET_HEIGHT;

    snapshotCanvas.width = width;
    snapshotCanvas.height = height;

    snapshotCtx.clearRect(0, 0, width, height);

    try {
        if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            snapshotCtx.save();
            snapshotCtx.translate(width, 0);
            snapshotCtx.scale(-1, 1);
            snapshotCtx.drawImage(video, 0, 0, width, height);
            snapshotCtx.restore();

            if (overlayCanvas.width && overlayCanvas.height) {
                snapshotCtx.drawImage(overlayCanvas, 0, 0, width, height);
            }
        } else {
            snapshotCtx.fillStyle = "#020617";
            snapshotCtx.fillRect(0, 0, width, height);
            snapshotCtx.fillStyle = "#e5e7eb";
            snapshotCtx.font = "bold 24px system-ui, sans-serif";
            snapshotCtx.fillText("No active video frame", 28, 56);
        }
    } catch (error) {
        snapshotCtx.fillStyle = "#020617";
        snapshotCtx.fillRect(0, 0, width, height);
        snapshotCtx.fillStyle = "#e5e7eb";
        snapshotCtx.font = "bold 24px system-ui, sans-serif";
        snapshotCtx.fillText("Snapshot unavailable", 28, 56);
    }

    return snapshotCanvas.toDataURL("image/jpeg", 0.65);
}

function purgeLogs() {
    lastViolationTimes = {};
    incidentTotal = 0;
    if (incidentCount) incidentCount.textContent = "0";

    const cards = logFeed.querySelectorAll(".incident-card");

    for (const card of cards) {
        card.remove();
    }

    if (emptyFeed) {
        emptyFeed.classList.remove("hidden");
    }

    if (feedStatus) {
        feedStatus.textContent = "🛡️ Environment Clear";
        feedStatus.style.color = "";
        feedStatus.style.borderColor = "";
        feedStatus.style.background = "";
    }
}

function exportViolationsAsZip() {
    const cards = Array.from(logFeed.querySelectorAll(".incident-card"));

    if (!cards.length) {
        if (feedStatus) feedStatus.textContent = "No violations available for export";
        return;
    }

    const files = cards.map((card, index) => {
        const snapshot = card.dataset.snapshot;
        if (!snapshot) return null;

        const type = (card.dataset.violationType || `violation-${index + 1}`).replace(/\s+/g, "_").toLowerCase();
        const timeLabel = card.querySelector(".incident-time")?.textContent.replace(/[: ]/g, "_") || `event_${index + 1}`;
        const filename = `violation_${index + 1}_${type}_${timeLabel}.jpg`;

        return {
            name: filename,
            data: base64ToUint8Array(snapshot)
        };
    }).filter(Boolean);

    if (!files.length) {
        if (feedStatus) feedStatus.textContent = "No snapshot data available to export";
        return;
    }

    const zipBlob = createZipBlob(files);
    const downloadName = `violations-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.zip`;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(zipBlob);
    link.download = downloadName;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function base64ToUint8Array(dataUrl) {
    const parts = dataUrl.split(",");
    const base64 = parts[1] || "";
    const binary = atob(base64);
    const buffer = new Uint8Array(binary.length);

    for (let i = 0; i < binary.length; i += 1) {
        buffer[i] = binary.charCodeAt(i);
    }

    return buffer;
}

function crc32(bytes) {
    let crc = -1;

    for (let i = 0; i < bytes.length; i += 1) {
        crc ^= bytes[i];
        for (let j = 0; j < 8; j += 1) {
            crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
        }
    }

    return (crc ^ -1) >>> 0;
}

function createZipBlob(files) {
    const encoder = new TextEncoder();
    const localEntries = [];
    const centralEntries = [];
    let offset = 0;

    for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const crc = crc32(file.data);
        const header = new Uint8Array(30 + nameBytes.length);
        const view = new DataView(header.buffer);

        view.setUint32(0, 0x04034b50, true);
        view.setUint16(4, 20, true);
        view.setUint16(6, 0, true);
        view.setUint16(8, 0, true);
        view.setUint16(10, 0, true);
        view.setUint32(14, crc, true);
        view.setUint32(18, file.data.length, true);
        view.setUint32(22, file.data.length, true);
        view.setUint16(26, nameBytes.length, true);
        header.set(nameBytes, 30);

        localEntries.push(header, file.data);

        const central = new Uint8Array(46 + nameBytes.length);
        const centralView = new DataView(central.buffer);

        centralView.setUint32(0, 0x02014b50, true);
        centralView.setUint16(4, 20, true);
        centralView.setUint16(6, 20, true);
        centralView.setUint16(8, 0, true);
        centralView.setUint16(10, 0, true);
        centralView.setUint16(12, 0, true);
        centralView.setUint32(16, crc, true);
        centralView.setUint32(20, file.data.length, true);
        centralView.setUint32(24, file.data.length, true);
        centralView.setUint16(28, nameBytes.length, true);
        centralView.setUint32(42, offset, true);
        central.set(nameBytes, 46);

        centralEntries.push(central);
        offset += header.length + file.data.length;
    }

    const directorySize = centralEntries.reduce((sum, entry) => sum + entry.length, 0);
    const endRecord = new Uint8Array(22);
    const endView = new DataView(endRecord.buffer);

    endView.setUint32(0, 0x06054b50, true);
    endView.setUint16(8, files.length, true);
    endView.setUint16(10, files.length, true);
    endView.setUint32(12, directorySize, true);
    endView.setUint32(16, offset, true);

    return new Blob([...localEntries, ...centralEntries, endRecord], { type: "application/zip" });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}