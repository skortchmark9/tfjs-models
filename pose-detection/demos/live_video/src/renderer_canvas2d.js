/**
 * @license
 * Copyright 2023 Google LLC.
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
import * as posedetection from '@tensorflow-models/pose-detection';
import * as scatter from 'scatter-gl';

import * as params from './params';
import { getKneePoints, getAngle2, calculateDistance } from './knee';
// import { demoData } from './demo_data';

// These anchor points allow the pose pointcloud to resize according to its
// position in the input.
const ANCHOR_POINTS = [[0, 0, 0], [0, 1, 0], [-1, 0, 0], [-1, -1, 0]];

// #ffffff - White
// #800000 - Maroon
// #469990 - Malachite
// #e6194b - Crimson
// #42d4f4 - Picton Blue
// #fabed4 - Cupid
// #aaffc3 - Mint Green
// #9a6324 - Kumera
// #000075 - Navy Blue
// #f58231 - Jaffa
// #4363d8 - Royal Blue
// #ffd8b1 - Caramel
// #dcbeff - Mauve
// #808000 - Olive
// #ffe119 - Candlelight
// #911eb4 - Seance
// #bfef45 - Inchworm
// #f032e6 - Razzle Dazzle Rose
// #3cb44b - Chateau Green
// #a9a9a9 - Silver Chalice
const COLOR_PALETTE = [
  '#ffffff', '#800000', '#469990', '#e6194b', '#42d4f4', '#fabed4', '#aaffc3',
  '#9a6324', '#000075', '#f58231', '#4363d8', '#ffd8b1', '#dcbeff', '#808000',
  '#ffe119', '#911eb4', '#bfef45', '#f032e6', '#3cb44b', '#a9a9a9'
];
export class RendererCanvas2d {
  constructor(canvas) {
    this.ctx = canvas.getContext('2d');
    this.canvas = canvas;
    this.videoWidth = canvas.width;
    this.videoHeight = canvas.height;

    this.tempCanvas = document.createElement("canvas");
    this.resize(canvas.width, canvas.height);

    this.tempCanvas.style = 'position: absolute; right: 0; top: 0;';
    this.tempCtx = this.tempCanvas.getContext("2d");

    const startX = 180;
    const startY = 290;
    const imageData = this.tempCtx.getImageData(0, 0, this.videoWidth, this.videoHeight);
    const pixels = imageData.data;
    const demoData = [];
    for (let i = 0; i < demoData.length; i++) {
      pixels[i] = demoData[i];
    }
    this.tempCtx.putImageData(imageData, 0, 0);


    const maskPixels = new Uint8ClampedArray(demoData.length);
    maskPixels.fill(0);
    // use pixels to figure out fringe
    this.regionGrow(imageData, maskPixels, startX, startY);
    for (let i = 0; i < maskPixels.length; i++) {
      if (maskPixels[i] !== 0) {
        pixels[i] = maskPixels[i];
      }
    }
    this.tempCtx.putImageData(imageData, 0, 0);

    // build fringe around knee point
    this.tempCtx.fillStyle = 'Red';
    this.tempCtx.strokeStyle = 'White';
    this.tempCtx.lineWidth = params.DEFAULT_LINE_WIDTH;

    const circle = new Path2D();
    circle.arc(startX, startY, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
    this.tempCtx.fill(circle);
    this.tempCtx.stroke(circle);

    // document.body.appendChild(this.tempCanvas);
  }

  resize(videoWidth, videoHeight) {
    this.videoWidth = videoWidth;
    this.videoHeight = videoHeight;

    this.canvas.width = this.videoWidth;
    this.canvas.height = this.videoHeight;

    this.tempCanvas.width = videoWidth;
    this.tempCanvas.height = videoHeight;

    // technically need to redraw but can rely on raf loop for now
  }

  draw(rendererParams) {
    const [video, poses, isModelChanged] = rendererParams;
    this.drawCtx(video);

    // The null check makes sure the UI is not in the middle of changing to a
    // different model. If during model change, the result is from an old model,
    // which shouldn't be rendered.
    if (poses && poses.length > 0 && !isModelChanged) {
      this.drawResults(poses);
    }
  }

  drawCtx(video) {
    this.ctx.drawImage(video, 0, 0, this.videoWidth, this.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.videoWidth, this.videoHeight);
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param poses A list of poses to render.
   */
  drawResults(poses) {
    for (const pose of poses) {
      this.drawResult(pose);
    }
  }

  /**
   * Draw the keypoints and skeleton on the video.
   * @param pose A pose with keypoints to render.
   */
  drawResult(pose) {
    if (pose.keypoints != null) {
      let kneepoints = getKneePoints(pose.keypoints);
      this.lastKneepoints = kneepoints;
      if (!kneepoints) {
        return;
      }
      this.drawKneeField(kneepoints, pose.id);
      this.drawKneepoints(kneepoints, pose.id);
      this.drawAngle(kneepoints, pose.id);
    }

    if (pose.keypoints3D != null && params.STATE.modelConfig.render3D) {
      this.drawKeypoints3D(pose.keypoints3D);
    }
  }

  drawKneepoints(kneepoints) {
    const drawLine = (kp1, kp2) => {
      this.ctx.beginPath();
      this.ctx.moveTo(kp1.x, kp1.y);
      this.ctx.lineTo(kp2.x, kp2.y);
      this.ctx.stroke();
    };

    this.ctx.fillStyle = 'Red';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    this.ctx.fillStyle = 'Green';
    for (const pt of kneepoints) {
      this.drawKeypoint(pt);
    }

    drawLine(kneepoints[0], kneepoints[1]);
    drawLine(kneepoints[1], kneepoints[2]);
  }

  calculateAngle(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const angle = Math.atan2(dy, dx);
    return angle;
  }

  drawAngle(kneepoints) {
    const ctx = this.ctx;
    // Assuming you have the keypoints for "right_foot", "right_knee", and "right_hip"
    const foot = kneepoints[2];
    const knee = kneepoints[1];
    const hip = kneepoints[0];

    // Calculate the knee angle using the two points (foot and knee)
    const kneeFootAngle = getAngle2(knee, foot);

    // Calculate the knee angle between the knee and hip points
    const kneeHipAngle = getAngle2(knee, hip);

    // Draw the arc overlay
    const centerX = knee.x;
    const centerY = knee.y;
    const radius = 50; // Adjust the radius of the arc as needed

    ctx.beginPath();
    // default to clockwise, but remember the whole scene is flipped
    let counterclockwise = true;
    if (foot.x < hip.x) {
      counterclockwise = false;
    }

    ctx.arc(centerX, centerY, radius, kneeFootAngle, kneeHipAngle + Math.PI, counterclockwise);
    ctx.strokeStyle = "#eae0e0";
    ctx.stroke();

    ctx.setLineDash([2, 5]);
    ctx.lineTo(knee.x, knee.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }


  colorDiff(p1, p2) {
    // Check if the pixel color is similar to the knee color
    const rDiff = Math.abs(p1.r - p2.r);
    const gDiff = Math.abs(p1.g - p2.g);
    const bDiff = Math.abs(p1.b - p2.b);

    return rDiff + gDiff + bDiff;
  }

  // Function to check if a pixel is part of the leg based on color similarity
  isLegPixel(imageData, kneeColor, x, y) {
    const pixelColor = this.getPixelColor(imageData, x, y);

    // Define a color similarity threshold (adjust as needed)
    const colorThreshold = 12;


    return rDiff < colorThreshold && gDiff < colorThreshold && bDiff < colorThreshold;
  }


  // Helper function to perform region growing
  regionGrow(imageData, maskPixels, startX, startY) {
    const kneeColor = this.getPixelColor(imageData, startX, startY);
    const visited = new Set();
    const stack = [{ x: startX, y: startY }];
    let count = 0;

    while (stack.length > 0 && count < 100000) {
      const { x, y } = stack.pop();

      const pixelColor = this.getPixelColor(imageData, x, y);
      const colorDiff = this.colorDiff(kneeColor, pixelColor);
      const { colorWeight, distanceWeight, fringeThreshold } = params.STATE.modelConfig;
      const distance = calculateDistance(startX, startY, x, y);
      let isLegPixel = false;
      if ((colorDiff * colorWeight) + (distance * distanceWeight) < fringeThreshold) {
        isLegPixel = true;
      }
      // if (distance > 200) {
      //   isLegPixel = false;
      // }

      if (!visited.has(`${x},${y}`) && isLegPixel) {
        visited.add(`${x},${y}`);
        count++;

        const index = (y * this.videoWidth + x) * 4;
        maskPixels[index + 1] = 255; // Set the green value to 255 (opaque) to include the pixel in the mask
        maskPixels[index + 3] = 100; // Set the alpha value to 255 (opaque) to include the pixel in the mask

        // Add neighboring pixels to the stack for further processing
        if (x + 1 < this.videoWidth) stack.push({ x: x + 1, y });
        if (x - 1 >= 0) stack.push({ x: x - 1, y });
        if (y + 1 < this.videoHeight) stack.push({ x, y: y + 1 });
        if (y - 1 >= 0) stack.push({ x, y: y - 1 });
      }
    }
  }

  // Function to get the color of a pixel at given coordinates (x, y)
  getPixelColor(imageData, x, y) {
    const index = (y * this.videoWidth + x) * 4;
    return {
      r: imageData.data[index + 0],
      g: imageData.data[index + 1],
      b: imageData.data[index + 2],
    };
  }


  drawKneeField(pts) {
    // Assuming you have the keypoints for "right_foot", "right_knee", and "right_hip"
    const footX = Math.round(pts[0].x);
    const footY = Math.round(pts[0].y);
    const kneeX = Math.round(pts[1].x);
    const kneeY = Math.round(pts[1].y);
    const hipX = Math.round(pts[2].x);
    const hipY = Math.round(pts[2].y);

    // Get the canvas element
    const ctx = this.ctx;
    this.tempCtx.clearRect(0, 0, this.videoWidth, this.videoHeight);

    // Create a mask to represent the leg region
    const maskData = this.tempCtx.getImageData(0, 0, this.videoWidth, this.videoHeight);
    const maskPixels = maskData.data;

    // Get the current pixel data
    const pixelData = ctx.getImageData(0, 0, this.videoWidth, this.videoHeight);

    // Draw debug overlay
    let debug = false;
    // debug = true;
    if (debug) {
      for (let row = 0; row < this.videoHeight; row++) {
        if ([1, 2, 3].includes(row % 9)) {
          for (let col = 0; col < this.videoWidth; col++) {
            const index = (row * this.videoWidth + col) * 4;
            maskPixels[index] = 255; // Set the red value to 255 (opaque) to include the pixel in the mask
            maskPixels[index + 3] = 255; // Set the alpha value to 100 (opaque) to include the pixel in the mask
          }
        }
      }
    }


    this.regionGrow(pixelData, maskPixels, kneeX, kneeY);
    this.regionGrow(pixelData, maskPixels, hipX, hipY);
    this.regionGrow(pixelData, maskPixels, footX, footY);

    // Apply the leg mask to the canvas
    this.tempCtx.putImageData(maskData, 0, 0);


    let alpha = this.ctx.globalAlpha;
    this.ctx.globalAlpha = 0.4;
    this.ctx.drawImage(this.tempCanvas, 0, 0);
    this.ctx.globalAlpha = alpha;
  }


  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints.
   */
  drawKeypoints(keypoints) {
    const keypointInd =
      posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.ctx.fillStyle = 'Red';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    for (const i of keypointInd.middle) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Green';
    for (const i of keypointInd.left) {
      this.drawKeypoint(keypoints[i]);
    }

    this.ctx.fillStyle = 'Orange';
    for (const i of keypointInd.right) {
      this.drawKeypoint(keypoints[i]);
    }
  }

  drawKeypoint(keypoint) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, params.DEFAULT_RADIUS, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints.
   */
  drawSkeleton(keypoints, poseId) {
    // Each poseId is mapped to a color in the color palette.
    const color = params.STATE.modelConfig.enableTracking && poseId != null ?
      COLOR_PALETTE[poseId % 20] :
      'White';
    this.ctx.fillStyle = color;
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    posedetection.util.getAdjacentPairs(params.STATE.model).forEach(([
      i, j
    ]) => {
      const kp1 = keypoints[i];
      const kp2 = keypoints[j];

      // If score is null, just show the keypoint.
      const score1 = kp1.score != null ? kp1.score : 1;
      const score2 = kp2.score != null ? kp2.score : 1;
      const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;

      if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
        this.ctx.beginPath();
        this.ctx.moveTo(kp1.x, kp1.y);
        this.ctx.lineTo(kp2.x, kp2.y);
        this.ctx.stroke();
      }
    });
  }

  drawKeypoints3D(keypoints) {
    const scoreThreshold = params.STATE.modelConfig.scoreThreshold || 0;
    const pointsData =
      keypoints.map(keypoint => ([-keypoint.x, -keypoint.y, -keypoint.z]));

    const dataset =
      new scatter.ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);

    const keypointInd =
      posedetection.util.getKeypointIndexBySide(params.STATE.model);
    this.scatterGL.setPointColorer((i) => {
      if (keypoints[i] == null || keypoints[i].score < scoreThreshold) {
        // hide anchor points and low-confident points.
        return '#ffffff';
      }
      if (i === 0) {
        return '#ff0000' /* Red */;
      }
      if (keypointInd.left.indexOf(i) > -1) {
        return '#00ff00' /* Green */;
      }
      if (keypointInd.right.indexOf(i) > -1) {
        return '#ffa500' /* Orange */;
      }
    });

    if (!this.scatterGLHasInitialized) {
      this.scatterGL.render(dataset);
    } else {
      this.scatterGL.updateDataset(dataset);
    }
    const connections = posedetection.util.getAdjacentPairs(params.STATE.model);
    const sequences = connections.map(pair => ({ indices: pair }));
    this.scatterGL.setSequences(sequences);
    this.scatterGLHasInitialized = true;
  }
}
