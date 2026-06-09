#!/usr/bin/env bash
# install_esp_surface_generator.sh
# Clone esp-surface-generator, install its Node dependencies, and put the
# stdin adapter in place.  Run from the ge-molsg repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# 1. Check node is available and meets the minimum version requirement
# --------------------------------------------------------------------------
if ! node --version &>/dev/null; then
    NODE_ERR=$(node --version 2>&1 || true)
    if command -v node &>/dev/null; then
        echo "ERROR: 'node' found at $(command -v node) but failed to execute:"
        echo "  $NODE_ERR"
        echo "The installed Node.js binary is likely incompatible with this system's glibc."
    else
        echo "ERROR: 'node' not found on PATH."
    fi
    echo "Install a compatible Node.js version via nvm:"
    echo "  \`nvm install 17 && nvm use 17\`"
    exit 1
fi

NODE_MAJOR=$(node -e "process.stdout.write(process.versions.node.split('.')[0])")
if [[ "$NODE_MAJOR" -lt 14 ]]; then
    echo "ERROR: Node.js >= 14 required (found $(node --version))."
    exit 1
fi

echo "Using Node.js $(node --version)  (npm $(npm --version))"

# --------------------------------------------------------------------------
# 2. Clone if not already present
# --------------------------------------------------------------------------
if [[ -d "esp-surface-generator/.git" ]]; then
    echo "esp-surface-generator already cloned — skipping clone."
else
    echo "Cloning AstexUK/esp-surface-generator..."
    git clone https://github.com/AstexUK/esp-surface-generator.git
fi

# --------------------------------------------------------------------------
# 3. Install npm dependencies
# --------------------------------------------------------------------------
echo "Installing npm dependencies..."
cd esp-surface-generator
npm install --silent
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# 4. Copy the stdin adapter
# --------------------------------------------------------------------------
echo "Copying utils/mesh_from_stdin.js -> esp-surface-generator/"
cp utils/mesh_from_stdin.js esp-surface-generator/mesh_from_stdin.js

echo ""
echo "Done. Test with:"
echo "  echo 'ATOM      1   C1 MOL _ 1       0.000   0.000   0.000  0.0000  1.7000' | node esp-surface-generator/mesh_from_stdin.js"
