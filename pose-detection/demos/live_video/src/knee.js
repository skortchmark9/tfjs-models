import { STATE } from './params';

export const RAD_TO_DEG = 180 / Math.PI;
export const DEG_TO_RAD = Math.PI / 180;

export const KNEE_SELECTION = {
  'AUTO': 'AUTO',
  'LEFT': 'LEFT',
  'RIGHT': 'RIGHT',
};

const _state_ = {
  selection: localStorage.getItem('knee-selection') || KNEE_SELECTION.LEFT
};

export const selectKnee = (selection) => {
  localStorage.setItem('knee-selection', selection);
  _state_.selection = selection;
}

export const getSelection = () => {
  return _state_.selection;
}

export const sumScores = (pts) => pts.reduce((base, acc) => base + acc.score, 0);

export function getKneePoints(keypoints) {
  if (!keypoints) {
    return;
  }
  const fmt = (pts) => pts.map((pt) => ({ ...pt, x: Math.round(pt.x), y: Math.round(pt.y) }));

  let left_knee_points = keypoints.filter((point) => {
    return (['left_hip', 'left_knee', 'left_ankle'].includes(point.name))
  });
  const left_score = sumScores(left_knee_points);
  const left_angle = getAngle3Deg(...left_knee_points);
  left_knee_points = fmt(left_knee_points);

  let right_knee_points = keypoints.filter((point) => {
    return (['right_hip', 'right_knee', 'right_ankle'].includes(point.name))
  });
  const right_score = sumScores(right_knee_points);
  const right_angle = getAngle3Deg(...right_knee_points);
  right_knee_points = fmt(right_knee_points);

  const threshold = STATE.modelConfig.scoreThreshold * 1.5;

  let pts = right_knee_points;
  let score = right_score;
  let angle = right_angle;
  if (_state_.selection === KNEE_SELECTION.LEFT) {
    pts = left_knee_points;
    score = left_score;
    angle = left_angle;
  };

  // Not a legit angle.
  if ((180 - angle) > 170) {
    return undefined;
  }

  const d1 = calculateDistance(pts[0].x, pts[0].y, pts[1].x, pts[1].y);
  const d2 = calculateDistance(pts[1].x, pts[1].y, pts[2].x, pts[2].y);
  // The calculated distances between the hip-knee and the knee-ankle should be
  // similar. If one is substantially smaller, it's probably combining points, so
  // throw it out.
  const distanceThreshold = .2;

  if (Math.min(d2, d1) / Math.max(d2, d1) < distanceThreshold) {
    return undefined;
  }

  if (score > threshold) {
    return pts;
  }
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

// Function to calculate the Euclidean distance between two points
export function calculateDistance(x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  return Math.sqrt(dx * dx + dy * dy);
}
