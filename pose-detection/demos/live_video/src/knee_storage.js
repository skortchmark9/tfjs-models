/**
 * {
 *     image,
 *     kneepoints,
 *     dimensions,
 *     confidence,
 *     angleDeg,
 *     displayAngle,
 *     date
 * }
 */
export function put(entry) {
  let db = localStorage.getItem('db');
  if (db) {
    db = JSON.parse(db);
  }
  if (!db) {
    db = []
  }

  const { image, displayAngle, date } = entry;
  const key = `${displayAngle}deg@${date}`;
  delete entry.image;

  db.push(entry);

  // Commit this item
  localStorage.setItem(key, image);

  // Add it to the DB
  localStorage.setItem('db', JSON.stringify(db));

  return key;
}

export function getImage(key) {
  const dataURL = localStorage.getItem(key);
  return dataURL;
}

export function getEntries() {
  let db = localStorage.getItem('db');
  if (db) {
    db = JSON.parse(db);
  }
  return db || [];
}
