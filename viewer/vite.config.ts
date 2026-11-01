import { createReadStream, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { extname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020';
import { defineConfig, type Plugin } from 'vite';

const SCENES_DIR = fileURLToPath(new URL('../scenes', import.meta.url));
const POIS_SCHEMA = fileURLToPath(new URL('../schemas/pois.schema.json', import.meta.url));
const MAX_POIS_BYTES = 1024 * 1024;
const CONTENT_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg',
  '.json': 'application/json',
  '.splat': 'application/octet-stream',
};

// Dev only: serves scenes/worlds.json and scenes/<name>/bundle/*. Nothing else under scenes/ is exposed.
function serveSceneBundles(): Plugin {
  return {
    name: 'atomspace-scene-bundles',
    configureServer(server) {
      server.middlewares.use('/scenes', (req, res, next) => {
        const urlPath = decodeURIComponent((req.url ?? '').split('?')[0]);
        const file = resolve(SCENES_DIR, '.' + urlPath);
        const parts = relative(SCENES_DIR, file).split(sep);
        const isWorldList = parts.length === 1 && parts[0] === 'worlds.json';
        if (!isWorldList && (parts.includes('..') || parts.length < 3 || parts[1] !== 'bundle')) return next();
        try {
          if (!statSync(file).isFile()) return next();
        } catch {
          return next();
        }
        res.setHeader('Content-Type', CONTENT_TYPES[extname(file)] ?? 'application/octet-stream');
        createReadStream(file).pipe(res);
      });
    },
  };
}

// Dev only: POST /api/scenes/<name>/pois replaces that scene's bundle/pois.json and nothing else.
function savePois(): Plugin {
  return {
    name: 'atomspace-save-pois',
    configureServer(server) {
      const validate = new Ajv2020({ strict: false }).compile(JSON.parse(readFileSync(POIS_SCHEMA, 'utf8')));

      server.middlewares.use('/api/scenes', (req, res) => {
        const fail = (status: number, message: string) => {
          res.statusCode = status;
          res.end(message);
        };

        const match = (req.url ?? '').split('?')[0].match(/^\/([A-Za-z0-9_-]+)\/pois$/);
        if (!match) return fail(404, 'Not found');
        if (req.method !== 'POST') return fail(405, 'Use POST');
        // Other sites can't send this content type to localhost without a preflight, which we never answer.
        if (req.headers['content-type'] !== 'application/json') return fail(415, 'Send application/json');

        const target = resolve(SCENES_DIR, match[1], 'bundle', 'pois.json');
        try {
          if (!statSync(resolve(target, '..')).isDirectory()) return fail(404, 'Unknown scene');
        } catch {
          return fail(404, 'Unknown scene');
        }

        const chunks: Buffer[] = [];
        let size = 0;
        let tooLarge = false;
        req.on('data', (chunk: Buffer) => {
          size += chunk.length;
          if (size > MAX_POIS_BYTES) tooLarge = true;
          else chunks.push(chunk);
        });
        req.on('end', () => {
          if (tooLarge) return fail(413, 'Too large');
          let body: unknown;
          try {
            body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
          } catch {
            return fail(400, 'Not valid JSON');
          }
          if (!validate(body)) return fail(400, `Does not match pois.schema.json: ${JSON.stringify(validate.errors)}`);
          writeFileSync(`${target}.tmp`, JSON.stringify(body, null, 2));
          renameSync(`${target}.tmp`, target);
          res.end('Saved');
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [serveSceneBundles(), savePois()],
});
