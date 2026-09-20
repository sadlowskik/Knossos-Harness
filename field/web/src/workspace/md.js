import MarkdownIt from 'markdown-it';
import sanitizeHtml from 'sanitize-html';

// Repository text remains untrusted inside an authenticated operator tab.
const markdown = new MarkdownIt({ html: false, linkify: false, typographer: false });

function safeHref(value) {
  if (!value || /["'`<>\\\u0000-\u0020\u007f]/.test(value)) return false;
  if (value.startsWith('//')) return false;
  if (/^(#|\/|\.\.?\/)/.test(value)) return true;
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password;
  } catch {
    return !value.includes(':');
  }
}

export function renderMarkdown(src) {
  return sanitizeHtml(markdown.render(String(src ?? '')), {
    allowedTags: ['p', 'br', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'strong', 'em', 's',
      'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td'],
    allowedAttributes: { a: ['href', 'title', 'target', 'rel'], ol: ['start'] },
    allowedSchemes: ['http', 'https'],
    allowProtocolRelative: false,
    transformTags: {
      a: (tagName, attrs) => ({
        tagName: safeHref(attrs.href) ? 'a' : 'span',
        attribs: safeHref(attrs.href)
          ? { ...attrs, target: '_blank', rel: 'noopener noreferrer' }
          : {},
      }),
    },
  }).trim();
}
