export function validateBrowserUrl(value, allowedDomains = []) {
  let parsed;
  try { parsed = new URL(String(value ?? '').trim()); } catch {
    return { ok: false, error: 'Enter a complete http:// or https:// URL.' };
  }
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    return { ok: false, error: 'Only HTTP and HTTPS browser routes are allowed.' };
  }
  if (parsed.username || parsed.password) {
    return { ok: false, error: 'Browser routes may not contain embedded credentials.' };
  }
  const allowed = new Set(allowedDomains.map((domain) => String(domain).toLowerCase()));
  if (!allowed.has(parsed.hostname.toLowerCase())) {
    return { ok: false, error: `${parsed.hostname} is not in field.yaml websites.` };
  }
  return { ok: true, url: parsed.href };
}
