import { get, set } from 'idb-keyval';
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

  const { image } = entry;
  const key = getKey(entry);
  delete entry.image;

  db.push(entry);

  // Commit this image
  set(key, image);

  // Add it to the DB
  localStorage.setItem('db', JSON.stringify(db));

  return key;
}

export function deleteEntry(index) {
  const db = getEntries();
  const entry = db[index];
  localStorage.removeItem(getKey(entry));
  db.splice(index, 1);

  localStorage.setItem('db', JSON.stringify(db));
  return db;
}

export function getKey(entry) {
  if (!entry) {
    throw new Error('No valid entry.');
  }
  const key = `${entry.displayAngle}deg@${entry.date}`;
  return key;
}

export async function getImage(keyOrEntry) {
  let key = keyOrEntry;
  if (typeof keyOrEntry !== 'string') {
    key = getKey(keyOrEntry);
  }

  const dataURL = get(key);
  return dataURL;
}

export function getEntries() {
  let db = localStorage.getItem('db');
  if (db) {
    db = JSON.parse(db);
  }
  return db || [];
}
