"use strict";

importScripts('/vendor/onnxruntime/ort.all.min.js');
let outputName = null;
let activeProvider = "wasm";

const CONFIDENCE_THRESHOLD = 0.25;
const NMS_THRESHOLD = 0.45;
const MODEL_INPUT_SIZE = 640;

const FORBIDDEN_CLASSES = new Set([63, 67]);

self.onmessage = async function (event) {
    const message = event.data || {};

    try {
        if (message.action === "INIT") {
            await initializeYolo(message.modelUrl);
            return;
        }

        if (message.action === "INFERENCE") {
            await runInference(message.tensorData, message.dims);
            return;
        }
    } catch (error) {
        self.postMessage({
            status: "ERROR",
            error: error && error.message ? error.message : String(error)
        });
    }
};

async function initializeYolo(modelUrl) {
    if (!modelUrl) {
        throw new Error("Missing YOLO model URL.");
    }

    ort.env.wasm.wasmPaths = "/vendor/onnxruntime/";
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;
    ort.env.logLevel = "warning";

    try {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["webgpu"],
            graphOptimizationLevel: "all"
        });

        activeProvider = "webgpu";
        inputName = session.inputNames[0];
        outputName = session.outputNames[0];

        self.postMessage({
            status: "READY_WEBGPU"
        });
    } catch (webgpuError) {
        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all"
        });

        activeProvider = "wasm";
        inputName = session.inputNames[0];
        outputName = session.outputNames[0];

        self.postMessage({
            status: "READY_WASM"
        });
    }
}

async function runInference(tensorData, dims) {
    if (!session) {
        throw new Error("YOLO session is not initialized.");
    }

    const inputTensor = new ort.Tensor("float32", tensorData, dims);
    const feeds = {};
    feeds[inputName] = inputTensor;

    const outputs = await session.run(feeds);
    const outputTensor = outputs[outputName] || outputs[Object.keys(outputs)[0]];

    const detections = decodeOutput(outputTensor);
    const flat = [];

    for (const detection of detections) {
        flat.push(
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.confidence,
            detection.classId
        );
    }

    self.postMessage({
        status: "RESULT",
        provider: activeProvider,
        data: flat
    });
}

function decodeOutput(outputTensor) {
    const data = outputTensor.data;
    const dims = outputTensor.dims || [];

    let boxes = [];

    if (looksLikeFlatSix(data, dims)) {
        boxes = parseFlatSix(data);
    } else {
        boxes = parseRawYolo(data, dims);
    }

    boxes = boxes.filter((box) => {
        return (
            box.confidence >= CONFIDENCE_THRESHOLD &&
            FORBIDDEN_CLASSES.has(box.classId)
        );
    });

    return nonMaxSuppression(boxes, NMS_THRESHOLD);
}

function looksLikeFlatSix(data, dims) {
    if (dims.length === 2 && dims[1] === 6) return true;
    if (dims.length === 3 && dims[2] === 6) return true;
    if (data.length % 6 === 0 && data.length <= 6000) return true;
    return false;
}

function parseFlatSix(data) {
    const detections = [];

    for (let i = 0; i + 5 < data.length; i += 6) {
        const x1 = clamp(Number(data[i]), 0, MODEL_INPUT_SIZE);
        const y1 = clamp(Number(data[i + 1]), 0, MODEL_INPUT_SIZE);
        const x2 = clamp(Number(data[i + 2]), 0, MODEL_INPUT_SIZE);
        const y2 = clamp(Number(data[i + 3]), 0, MODEL_INPUT_SIZE);
        const confidence = Number(data[i + 4]);
        const classId = Math.round(Number(data[i + 5]));

        if (!Number.isFinite(confidence)) continue;
        if (x2 <= x1 || y2 <= y1) continue;

        detections.push({
            x1,
            y1,
            x2,
            y2,
            confidence,
            classId
        });
    }

    return detections;
}

function parseRawYolo(data, dims) {
    const detections = [];

    let attributes = 0;
    let candidates = 0;
    let layout = "NCH";

    if (dims.length === 3) {
        const a = dims[1];
        const b = dims[2];

        if (a <= 256 && b > a) {
            attributes = a;
            candidates = b;
            layout = "ATTR_FIRST";
        } else {
            candidates = a;
            attributes = b;
            layout = "BOX_FIRST";
        }
    } else if (dims.length === 2) {
        candidates = dims[0];
        attributes = dims[1];
        layout = "BOX_FIRST";
    } else {
        return detections;
    }

    if (attributes < 6 || candidates < 1) {
        return detections;
    }

    const hasObjectness = attributes >= 85;
    const classStart = hasObjectness ? 5 : 4;

    for (let i = 0; i < candidates; i++) {
        const cx = getYoloValue(data, layout, attributes, candidates, i, 0);
        const cy = getYoloValue(data, layout, attributes, candidates, i, 1);
        const w = getYoloValue(data, layout, attributes, candidates, i, 2);
        const h = getYoloValue(data, layout, attributes, candidates, i, 3);

        const objectness = hasObjectness
            ? getYoloValue(data, layout, attributes, candidates, i, 4)
            : 1.0;

        let bestClass = -1;
        let bestClassScore = 0;

        for (let c = classStart; c < attributes; c++) {
            const score = getYoloValue(data, layout, attributes, candidates, i, c);
            if (score > bestClassScore) {
                bestClassScore = score;
                bestClass = c - classStart;
            }
        }

        const confidence = objectness * bestClassScore;

        if (confidence < CONFIDENCE_THRESHOLD) continue;
        if (!FORBIDDEN_CLASSES.has(bestClass)) continue;

        let boxCx = Number(cx);
        let boxCy = Number(cy);
        let boxW = Number(w);
        let boxH = Number(h);

        if (
            Math.abs(boxCx) <= 2 &&
            Math.abs(boxCy) <= 2 &&
            Math.abs(boxW) <= 2 &&
            Math.abs(boxH) <= 2
        ) {
            boxCx *= MODEL_INPUT_SIZE;
            boxCy *= MODEL_INPUT_SIZE;
            boxW *= MODEL_INPUT_SIZE;
            boxH *= MODEL_INPUT_SIZE;
        }

        const x1 = clamp(boxCx - boxW / 2, 0, MODEL_INPUT_SIZE);
        const y1 = clamp(boxCy - boxH / 2, 0, MODEL_INPUT_SIZE);
        const x2 = clamp(boxCx + boxW / 2, 0, MODEL_INPUT_SIZE);
        const y2 = clamp(boxCy + boxH / 2, 0, MODEL_INPUT_SIZE);

        if (x2 <= x1 || y2 <= y1) continue;

        detections.push({
            x1,
            y1,
            x2,
            y2,
            confidence,
            classId: bestClass
        });
    }

    return detections;
}

function getYoloValue(data, layout, attributes, candidates, boxIndex, attrIndex) {
    if (layout === "ATTR_FIRST") {
        return data[attrIndex * candidates + boxIndex];
    }

    return data[boxIndex * attributes + attrIndex];
}

function nonMaxSuppression(boxes, threshold) {
    const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selected = [];

    while (sorted.length) {
        const current = sorted.shift();
        selected.push(current);

        for (let i = sorted.length - 1; i >= 0; i--) {
            if (sorted[i].classId !== current.classId) continue;

            const overlap = iou(current, sorted[i]);
            if (overlap > threshold) {
                sorted.splice(i, 1);
            }
        }
    }

    return selected;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
    const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
    const union = areaA + areaB - intersection;

    if (union <= 0) return 0;
    return intersection / union;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

let session = null;
let inputName = null;