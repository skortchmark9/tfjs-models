import { STATE } from './params';

export const RAD_TO_DEG = 180 / Math.PI;
export const DEG_TO_RAD = Math.PI / 180;

export const sumScores = (pts) => pts.reduce((base, acc) => base + acc.score, 0);

export function getKneePoints(keypoints) {
  if (!keypoints) {
    return;
  }

  const left_knee_points = keypoints.filter((point) => {
    return (['left_hip', 'left_knee', 'left_ankle'].includes(point.name))
  });

  const right_knee_points = keypoints.filter((point) => {
    return (['right_hip', 'right_knee', 'right_ankle'].includes(point.name))
  });

  let pts = left_knee_points;
  if (sumScores(left_knee_points) < sumScores(right_knee_points)) {
    pts = right_knee_points;
  }

  if (sumScores(pts) < (STATE.modelConfig.scoreThreshold * 3.5)) {
    return;
  }

  return pts.map((pt) => ({ ...pt, x: Math.round(pt.x), y: Math.round(pt.y) }));

}

export function getAngle2(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const angle = Math.atan2(dy, dx);
  return angle;
}

function getAngle3(point1, point2, point3) {
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
  return angleRad;
}

export function getAngle3Deg(p1, p2, p3) {
  return getAngle3(p1, p2, p3) * RAD_TO_DEG;
}
