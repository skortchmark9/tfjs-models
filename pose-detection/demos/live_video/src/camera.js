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
import * as params from './params';
import { isLandscape, isMobile } from './util';


export class Camera {
  constructor() {
    this.video = document.getElementById('video');
  }

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static async setup(cameraParam) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const { targetFPS, sizeOption } = cameraParam;
    let size = params.VIDEO_SIZE[sizeOption];
    let { width, height } = size;
    // Only setting the video to a specified size for large screen, on
    // mobile devices accept the default size.
    if (isLandscape()) {
      // size = params.VIDEO_SIZE['640 X 360'];
      width = window.innerWidth * (2 / 3);
      height = window.innerHeight;
    } else {
      // Note this does the wrong thing when switching
      // BACK to portrait mode, for mysterious reasons.
      // https://developer.apple.com/forums/thread/717988
      width = window.innerWidth;
      height = window.innerWidth;
    }

    const videoConfig = {
      'audio': false,
      'video': {
        facingMode: 'user',
        width,
        height,
        frameRate: {
          ideal: targetFPS,
        }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(videoConfig);

    const camera = new Camera();
    camera.video.srcObject = stream;

    await new Promise((resolve) => {
      camera.video.onloadedmetadata = (evt) => {
        resolve(video);
      };
    });

    camera.video.play();

    const videoWidth = camera.video.videoWidth;
    const videoHeight = camera.video.videoHeight;
    console.log(`landscape = ${isLandscape()} vw = ${videoWidth} vh = ${videoHeight}`);
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    return camera;
  }
}
