/* One copy of the presentation settings, shared.

   The Board and Rome each used to hold their own useState(loadFieldSettings) and each
   ran `useEffect(() => saveFieldSettings(settings))`, so mounting one screen could write
   back what the other had just changed. This is a tiny external store over the same
   localStorage blob: read once, written once, and every screen sees the same object. */

import { useSyncExternalStore } from 'react';
import { applyTypeface, loadFieldSettings, normalizeFieldSettings, saveFieldSettings } from './fieldPreferences.js';

let current = loadFieldSettings();
const listeners = new Set();
let typefaceApplied = false;

function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
function snapshot() { return current; }

/** Replace the settings (value or updater), persist once, and notify every screen. */
export function setFieldSettings(next) {
  const value = typeof next === 'function' ? next(current) : next;
  const normalized = saveFieldSettings(value);
  if (JSON.stringify(normalized) === JSON.stringify(current)) return current;
  current = normalized;
  for (const fn of listeners) fn();
  return current;
}

/** Merge a patch into the settings. */
export function patchFieldSettings(patch) {
  return setFieldSettings((settings) => normalizeFieldSettings({ ...settings, ...patch }));
}

/* The typeface choice drives --font-sans. A stored custom blob that no longer parses
   quietly reverts to Plex rather than leaving the app on the fallback stack. This runs
   once per change, for the whole app, rather than once per mounted screen. */
function reconcileTypeface(settings) {
  applyTypeface(settings).then((effective) => {
    if (effective !== current.typeface) patchFieldSettings({ typeface: effective });
  }).catch(() => {});
}

export default function useFieldSettings() {
  return [useSyncExternalStore(subscribe, snapshot, snapshot), setFieldSettings];
}

// Applied once at module load, and again whenever the operator changes the typeface.
let lastTypeface = current.typeface;
let lastFont = current.customFont?.dataUrl ?? '';
if (typeof document !== 'undefined' && !typefaceApplied) {
  typefaceApplied = true;
  reconcileTypeface(current);
  subscribe(() => {
    const nextFont = current.customFont?.dataUrl ?? '';
    if (current.typeface === lastTypeface && nextFont === lastFont) return;
    lastTypeface = current.typeface;
    lastFont = nextFont;
    reconcileTypeface(current);
  });
}
