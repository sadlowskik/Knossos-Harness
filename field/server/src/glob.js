// Glob matching for watch-ignore lists and routine triggers.
//
// This is written as a single tokenizing pass on purpose. A chain of .replace() calls
// is the obvious way to write it and it is wrong: each replacement inserts regex
// syntax (`(?:`, `.*`) that later replacements then rewrite, so `**/` silently
// degrades into a pattern that matches nothing. Compile once, character by character.

/** Compile one glob into an anchored RegExp. `/` matches either path separator. */
export function globToRegExp(glob) {
  let out = '';
  let i = 0;
  while (i < glob.length) {
    const c = glob[i];

    if (c === '*') {
      if (glob[i + 1] === '*') {
        if (glob[i + 2] === '/') { out += '(?:.*[\\\\/])?'; i += 3; }   // **/  → any depth
        else { out += '.*'; i += 2; }                                    // **   → anything
      } else {
        out += '[^\\\\/]*'; i += 1;                                      // *    → one segment
      }
      continue;
    }
    if (c === '?') { out += '[^\\\\/]'; i += 1; continue; }
    if (c === '/') { out += '[\\\\/]'; i += 1; continue; }
    if ('.+^${}()|[]\\'.includes(c)) { out += '\\' + c; i += 1; continue; }
    out += c; i += 1;
  }
  return new RegExp('^' + out + '$');
}

/**
 * Build a matcher from a list of globs.
 *
 * A pattern like `**\/node_modules/**` only describes the directory's *contents*. The
 * watcher also asks about the directory itself, and answering "not ignored" there makes
 * it descend into a tree we just said to skip. So each `/**` pattern also contributes a
 * form matching the directory node.
 */
export function buildMatcher(patterns) {
  const res = [];
  for (const p of patterns) {
    res.push(globToRegExp(p));
    if (p.endsWith('/**')) res.push(globToRegExp(p.slice(0, -3)));
  }
  return (candidate) => res.some((r) => r.test(candidate));
}
