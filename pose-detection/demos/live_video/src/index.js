/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';

import * as mpPose from '@mediapipe/pose';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs-core';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import { Camera } from './camera';
import { RendererWebGPU } from './renderer_webgpu';
import { RendererCanvas2d } from './renderer_canvas2d';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setupStats } from './stats_panel';
import { setBackendAndEnvFlags } from './util';

console.log('wut?');

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let renderer = null;
let useGpuRenderer = false;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath:
            `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
          STATE.model, { runtime, modelType: STATE.modelConfig.type });
      }
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = { modelType };

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      return posedetection.createDetector(STATE.model, modelConfig);
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setupCamera(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats?.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (camera.video.readyState < 2) {
    await new Promise((resolve) => {
      camera.video.onloadeddata = () => {
        resolve(video);
      };
    });
  }

  let poses = null;
  let canvasInfo = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    if (useGpuRenderer && STATE.model !== 'PoseNet') {
      throw new Error('Only PoseNet supports GPU renderer!');
    }
    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      if (useGpuRenderer) {
        const [posesTemp, canvasInfoTemp] = await detector.estimatePosesGPU(
          camera.video,
          { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false },
          true);
        poses = posesTemp;
        canvasInfo = canvasInfoTemp;
      } else {
        poses = await detector.estimatePoses(
          camera.video,
          { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
      }
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    endEstimatePosesStats();
  }
  const rendererParams = useGpuRenderer ?
    [camera.video, poses, canvasInfo, STATE.modelConfig.scoreThreshold] :
    [camera.video, poses, STATE.isModelChanged];
  renderer.draw(rendererParams);
  drawKneeAngles(poses[0])
}
function calculateAngle(point1, point2, point3) {
  // Calculate the vectors between the points
  const vector1 = [point2.x - point1.x, point2.y - point1.y];
  const vector2 = [point2.x - point3.x, point2.y - point3.y];

  // Calculate the dot product of the two vectors
  const dotProduct = vector1[0] * vector2[0] + vector1[1] * vector2[1];

  // Calculate the magnitudes of the vectors
  const magnitude1 = Math.sqrt(vector1[0] ** 2 + vector1[1] ** 2);
  const magnitude2 = Math.sqrt(vector2[0] ** 2 + vector2[1] ** 2);

  // Calculate the cosine of the angle using the dot product and magnitudes
  const cosineAngle = dotProduct / (magnitude1 * magnitude2);

  // Calculate the angle in radians
  const angleRad = Math.acos(cosineAngle);

  // Convert the angle from radians to degrees
  const angleDeg = (180 / Math.PI) * angleRad;

  return angleDeg;
}

function sumScores(pts) {
  let total = 0;
  for (const pt of pts) {
    total += pt.score
  }
  return total;
}

/**
 * Draw the keypoints on the video.
 * @param keypoints A list of keypoints.
 */
function drawKneeAngles(pose) {
  if (!pose) {
    return
  }
  const keypoints = pose.keypoints;
  const left_knee_points = keypoints.filter((point) => {
    return (['left_hip', 'left_knee', 'left_ankle'].includes(point.name))
  });
  const left_score = sumScores(left_knee_points);

  const right_knee_points = keypoints.filter((point) => {
    return (['right_hip', 'right_knee', 'right_ankle'].includes(point.name))
  });
  const right_score = sumScores(right_knee_points);
  const threshold = 1;

  // Output
  let outputDiv = document.getElementById('output-angle');
  if (!outputDiv) {
    const stats = document.getElementById('stats');
    outputDiv = document.createElement('div');
    outputDiv.id = 'output-angle';
    outputDiv.style = 'font-size: 40px';
    stats.insertBefore(outputDiv, stats.firstChild)
  }

  const bestScore = Math.max(left_score, right_score);
  let pts;
  if (right_score == bestScore) {
    pts = right_knee_points;
  } else {
    pts = left_knee_points;
  }
  const angle = calculateAngle(...pts)

  if (bestScore > threshold) {
    const displayAngle = 180 - Math.round(angle);
    outputDiv.innerHTML = `${displayAngle}deg / ${bestScore.toFixed(3)}`;
  }
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    urlParams.set('model', 'movenet');
  }
  await setupDatGui(urlParams);

  // stats = setupStats();
  const isWebGPU = STATE.backend === 'tfjs-webgpu';
  const importVideo = (urlParams.get('importVideo') === 'true') && isWebGPU;

  camera = await Camera.setup(STATE.camera);

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  detector = await createDetector();
  const canvas = document.getElementById('output');
  canvas.width = camera.video.width;
  canvas.height = camera.video.height;
  useGpuRenderer = (urlParams.get('gpuRenderer') === 'true') && isWebGPU;
  if (useGpuRenderer) {
    renderer = new RendererWebGPU(canvas, importVideo);
  } else {
    renderer = new RendererCanvas2d(canvas);
  }

  const pauseBtn = document.getElementById('pause');
  let playing = true;
  pauseBtn.addEventListener('click', () => {
    playing = !playing;
    if (!playing) {
      cancelAnimationFrame(rafId);
    } else {
      renderPrediction();
    }
  });

  renderPrediction();

  screen.orientation.lock('landscape');

  // Listen for orientation changes
  // let wasLandscape = screen.orientation.type.startsWith('landscape');
  // window.addEventListener("orientationchange", function () {
  //   // Announce the new orientation number
  //   const isLandscape = screen.orientation.type.startsWith('landscape');
  //   STATE.isSizeOptionChanged = isLandscape && !wasLandscape;
  //   wasLandscape = isLandscape;
  // }, false);
};

app();

if (useGpuRenderer) {
  renderer.dispose();
}
