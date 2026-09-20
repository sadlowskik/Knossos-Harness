const DEFAULT_MAX_BYTES = 4 * 1024 * 1024;
const DEFAULT_TIMEOUT_MS = 10_000;

export class HttpInputError extends Error {
  constructor(message, { status = 400, code = 'bad_request' } = {}) {
    super(message);
    this.name = 'HttpInputError';
    this.status = status;
    this.code = code;
  }
}

/** Read one bounded JSON request body using wire bytes, not JavaScript string length. */
export function readJsonBody(req, {
  maxBytes = DEFAULT_MAX_BYTES,
  timeoutMs = DEFAULT_TIMEOUT_MS,
} = {}) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let bytes = 0;
    let settled = false;

    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      req.off('data', onData);
      req.off('end', onEnd);
      req.off('error', onError);
      req.off('aborted', onAborted);
      fn(value);
    };
    const fail = (error) => {
      req.resume?.();
      finish(reject, error);
    };
    const onData = (chunk) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      bytes += buffer.length;
      if (bytes > maxBytes) {
        fail(new HttpInputError('request body exceeds the allowed size', {
          status: 413,
          code: 'body_too_large',
        }));
        return;
      }
      chunks.push(buffer);
    };
    const onEnd = () => {
      if (bytes === 0) return finish(resolve, {});
      try {
        const value = JSON.parse(Buffer.concat(chunks, bytes).toString('utf8'));
        finish(resolve, value);
      } catch {
        fail(new HttpInputError('invalid JSON body', { code: 'invalid_json' }));
      }
    };
    const onError = () => fail(new HttpInputError('request body could not be read'));
    const onAborted = () => fail(new HttpInputError('request body was aborted', {
      status: 400,
      code: 'body_aborted',
    }));
    const timer = setTimeout(() => fail(new HttpInputError('request body timed out', {
      status: 408,
      code: 'body_timeout',
    })), timeoutMs);

    req.on('data', onData);
    req.on('end', onEnd);
    req.on('error', onError);
    req.on('aborted', onAborted);
  });
}
