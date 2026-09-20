import assert from 'node:assert/strict';
import { renderMarkdown } from '../../web/src/workspace/md.js';

for (const source of [
  '<img src=x onerror=alert(1)>', '<svg><script>alert(1)</script></svg>',
  '[run](javascript:alert(1))', '[run](jav&#x61;script:alert(1))',
  '[run](data:text/html,hello)', '[share](//evil.example/path)',
  '[share](/\\evil.example/path)', '[mail](mailto:operator@example.test)',
  '[break](https://example.test/"onmouseover="alert(1))',
  '[secret](https://user:password@example.test/)',
  '![tracker](https://evil.example/pixel)',
  '<iframe srcdoc="<script>alert(1)</script>"></iframe>',
]) {
  const html = renderMarkdown(source);
  assert.doesNotMatch(html, /<(?:script|img|svg|iframe|style)\b/i, source);
  assert.doesNotMatch(html, /<[^>]+\son\w+\s*=/i, source);
  assert.doesNotMatch(html, /href="(?:javascript:|data:|mailto:|\/\/|\/\\)/i, source);
  assert.doesNotMatch(html, /href="https:\/\/user:password/i, source);
}
for (const href of ['https://example.test/docs?a=1&b=2', '#release-gates', '../docs/guide.md', 'guide.md']) {
  const html = renderMarkdown(`[safe](${href})`);
  assert.match(html, /<a href=/);
  assert.match(html, /rel="noopener noreferrer"/);
}
assert.match(renderMarkdown('```html\n<script>bad</script>\n```'), /<pre><code>&lt;script&gt;bad&lt;\/script&gt;/);
assert.match(renderMarkdown('1. parent\n   - child'), /<ol>[\s\S]*<ul>[\s\S]*child/);
assert.match(renderMarkdown('[reference][doc]\n\n[doc]: https://example.test'), /href="https:\/\/example.test"/);
console.log('markdown: CommonMark parsing and strict HTML/URL sanitization passed');
