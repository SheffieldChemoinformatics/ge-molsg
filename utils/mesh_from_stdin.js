/**
 * mesh_from_stdin.js
 *
 * Reads PQR from stdin, generates surface mesh, writes mesh to stdout.
 * Fix: ensure File/Blob/FileReader globals match jsdom window so NGL's instanceof
 * checks route to FileStreamer instead of NetworkStreamer.
 */

require('browser-env')();

// Make constructors globally consistent for NGL bundle
global.window = window;
global.document = document;
global.navigator = navigator;

global.File = window.File;
global.Blob = window.Blob;
global.FileReader = window.FileReader;

const fs = require('fs').promises;
const os = require('os');
const path = require('path');

// Load NGL bundle after globals are set
require('./ngl.dev.js');
const NGL = window.NGL || require('./ngl.dev.js');

if (!NGL || !NGL.autoLoad) throw new Error('NGL.autoLoad not found');

const SURF_PARAMS = {
  type: 'av',
  contour: false,
  scaleFactor: 4.0,
  probeRadius: 1.4,
  radiusParams: { type: 'explicit' },
  smooth: 0
};

function readStdin() {
  return new Promise((resolve, reject) => {
    let s = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', c => (s += c));
    process.stdin.on('end', () => resolve(s));
    process.stdin.on('error', reject);
  });
}

async function loadPqrTextViaTempFile(pqrText) {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ngl-pqr-'));
  const pqrPath = path.join(tmpDir, 'input.pqr');
  await fs.writeFile(pqrPath, pqrText, 'utf8');

  try {
    const data = await fs.readFile(pqrPath);

    // Use global File (same as window.File; helps instanceof inside bundle)
    const f = new File([data], 'input.pqr', { type: 'text/plain' });

    // ext hint
    return await NGL.autoLoad(f, { ext: 'pqr' });
  } finally {
    try { await fs.unlink(pqrPath); } catch {}
    try { await fs.rmdir(tmpDir); } catch {}
  }
}

(async () => {
  try {
    const pqrText = await readStdin();
    if (!pqrText || !pqrText.trim()) {
      console.error('No PQR received on stdin');
      process.exit(2);
    }

    const structure = await loadPqrTextViaTempFile(pqrText);

    const molsurf = new NGL.MolecularSurface(structure);
    const surf = molsurf.getSurface(SURF_PARAMS);

    const writer = new NGL.TMeshWriter(surf, { normals: false, precision: 5 });
    process.stdout.write(writer.getData());
  } catch (e) {
    console.error(e && e.stack ? e.stack : String(e));
    process.exit(1);
  }
})();
