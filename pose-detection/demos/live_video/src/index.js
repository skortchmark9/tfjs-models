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
import * as KneeStorage from './knee_storage';

tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';

import { Camera } from './camera';
import { RendererWebGPU } from './renderer_webgpu';
import { RendererCanvas2d } from './renderer_canvas2d';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setupStats } from './stats_panel';
import { getRelativeTime, isLandscape, setBackendAndEnvFlags } from './util';
import { getKneePoints, getAngle3Deg, sumScores, getSelection, selectKnee, KNEE_SELECTION } from './knee';

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
    console.time('camera setup');
    camera = await Camera.setup(STATE.camera);
    console.timeEnd('camera setup');
    STATE.isTargetFPSChanged = false;
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
      console.time('create detector');
      detector = await createDetector(STATE.model);
      console.timeEnd('create detector');
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }

  if (STATE.isSizeOptionChanged) {
    if (!useGpuRenderer) {
      renderer.resize(camera.video.width, camera.video.height);
    }
  }
  STATE.isSizeOptionChanged = false;
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
  drawScores(poses[0])
}


/**
 * Draw the keypoints on the video.
 * @param keypoints A list of keypoints.
 */
function drawScores(pose) {
  const pts = getKneePoints(pose?.keypoints);
  if (!pts) {
    return;
  }
  const score = sumScores(pts);

  // Output
  const outputAngle = document.getElementById('output-angle');
  const outputConfidence = document.getElementById('output-confidence');

  const angleDeg = getAngle3Deg(...pts);
  const displayAngle = 180 - Math.round(angleDeg);
  outputAngle.innerText = `${displayAngle}deg`;
  outputConfidence.innerText = `${score.toFixed(3)}`;
}

async function renderPrediction() {
  await checkGuiUpdate();

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(renderPrediction);
};

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    urlParams.set('model', 'movenet');
  }

  const sideCheckbox = document.getElementById('side-selector-checkbox');
  sideCheckbox.checked = getSelection() === KNEE_SELECTION.RIGHT;
  sideCheckbox.addEventListener('change', (evt) => {
    selectKnee(sideCheckbox.checked ? KNEE_SELECTION.RIGHT : KNEE_SELECTION.LEFT);
    return;
  });

  displayHistory();

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

  // Buttons
  const snapBtn = document.getElementById('snap');
  const delayedSnapBtn = document.getElementById('delayed-snap');
  const saveBtn = document.getElementById('save');
  const flipBtn = document.getElementById('flip-camera');

  let playing = true;
  const onSnap = () => {
    playing = !playing;
    snapBtn.innerHTML = playing ? `Snap` : `Retake`;
    if (!playing) {
      cancelAnimationFrame(rafId);
      saveBtn.removeAttribute('disabled');
    } else {
      renderPrediction();
      saveBtn.setAttribute('disabled', '');
    }
  };
  snapBtn.addEventListener('click', onSnap);

  let delayedSnapId;
  delayedSnapBtn.addEventListener('click', () => {
    if (delayedSnapId) {
      // Cancel
      clearInterval(delayedSnapId);
      delayedSnapId = null;
      delayedSnapBtn.innerText = `In 5s`;
      return;
    }

    let i = 6;
    const tick = () => {
      i--;
      delayedSnapBtn.innerText = `In ${i}s...`;
      if (i === 0) {
        onSnap();
        delayedSnapBtn.innerText = `In 5s`;
        clearInterval(delayedSnapId);
        delayedSnapId = null;
      }
    };

    delayedSnapId = setInterval(tick, 1000);
    tick();

    if (!playing) {
      renderPrediction();
    }
  });

  saveBtn.addEventListener('click', () => {
    const image = renderer.canvas.toDataURL();
    const kneepoints = renderer.lastKneepoints;
    if (!kneepoints) {
      alert('Never recorded a knee angle :(');
      return;
    }
    const { width, height } = renderer.canvas;
    const dimensions = { width, height };
    const confidence = sumScores(kneepoints);
    const angleDeg = getAngle3Deg(...kneepoints);
    const displayAngle = 180 - Math.round(angleDeg);
    const date = new Date().toISOString();

    const data = {
      image, kneepoints, dimensions, confidence, angleDeg, displayAngle, date
    };

    const key = KneeStorage.put(data);
    console.log('saved', key);
    displayHistory();
  });

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === 'videoinput');
  if (cameras.length > 1) {
    flipBtn.addEventListener('click', () => {
      const facingMode = STATE.camera.facingMode === 'user' ? 'environment' : 'user';
      STATE.camera.facingMode = facingMode;
      STATE.isSizeOptionChanged = true;
    });
  } else {
    flipBtn.style.display = 'none';
  }

  console.time('time to first framez');
  await renderPrediction();
  console.timeEnd('time to first framez');

  document.querySelector('.canvas-wrapper').classList.add('has-video');
  setTimeout(() => {
    document.getElementById('intro-placeholder').classList.add('fadeout');
  }, 2500);


  // Listen for orientation changes
  let wasLandscape = isLandscape();
  screen.orientation.addEventListener('change', function () {
    // Announce the new orientation number
    STATE.isSizeOptionChanged = isLandscape() !== wasLandscape;

    wasLandscape = isLandscape();
  }, false);
};

function displayHistory() {
  // DB
  const history = document.getElementById('history');
  const wrapper = document.querySelector('.canvas-wrapper')
  history.innerHTML = '';
  const entries = KneeStorage.getEntries();
  entries.reverse();
  if (!entries?.length) {
    return;
  }
  const max = Math.max(...entries.map((e) => e.displayAngle));
  const fragment = document.createDocumentFragment();

  const select = document.createElement('select');
  fragment.appendChild(select);

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.innerText = `View Snapshots - Record: ${max}°`;
  placeholder.setAttribute('disabled', '');
  placeholder.setAttribute('selected', '');
  select.appendChild(placeholder);


  entries.map((entry) => {
    const option = document.createElement('option');
    const { displayAngle, date } = entry;

    const isRecord = displayAngle === max;

    option.innerText = `${displayAngle}° - ${getRelativeTime(date)}${isRecord ? ' 🌟' : ''}`;
    select.appendChild(option);
  })

  select.addEventListener('change', (evt) => {
    const { selectedIndex } = evt.target.options;
    const img = document.getElementById('preview-image');
    // Special case - resume video
    if (selectedIndex === 0) {
      placeholder.innerText = `View Snapshots - Record: ${max}°`;
      placeholder.setAttribute('disabled', '');
      img.removeAttribute('src');
      wrapper.classList.remove('viewing-snapshot');
      return;
    }

    const idx = selectedIndex - 1;
    const entry = entries[idx];
    img.src = KneeStorage.getImage(entry);
    wrapper.classList.add('viewing-snapshot');
    placeholder.innerText = 'Resume Measurement'
    placeholder.removeAttribute('disabled');
  });

  history.appendChild(fragment);

}

app();

if (useGpuRenderer) {
  renderer.dispose();
}
