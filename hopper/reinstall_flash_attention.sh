#!/bin/bash

# Reinstall Flash Attention with DCP support
# This script will build and install the modified Flash Attention package

set -e  # Exit on any error

echo "🚀 Reinstalling Flash Attention with DCP support..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to Flash Attention root directory
cd "$FLASH_ATTN_ROOT"

echo "📍 Current directory: $(pwd)"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Set build environment variables
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
#export MAX_JOBS=4  # Adjust based on your system

# Uninstall existing flash-attn if present
echo "🗑️ Uninstalling existing flash-attn..."
pip uninstall flash-attn -y || true

# Build and install
echo "🔨 Building and installing Flash Attention..."
pip install -e . --no-build-isolation

# Verify installation
echo "✅ Verifying installation..."
python -c "
import flash_attn
print(f'Flash Attention version: {flash_attn.__version__}')

# Test DCP parameter support
from flash_attn import flash_attn_func
import inspect
sig = inspect.signature(flash_attn_func)
if 'cp_world_size' in sig.parameters and 'cp_rank' in sig.parameters:
    print('✅ DCP parameters found in flash_attn_func')
else:
    print('❌ DCP parameters NOT found in flash_attn_func')
    exit(1)

print('🎉 Flash Attention with DCP support installed successfully!')
"

echo "
🎯 Installation complete!

Usage example:
from flash_attn import flash_attn_func

# Standard usage (no DCP)
out = flash_attn_func(q, k, v, causal=True)

# DCP usage
out = flash_attn_func(q, k, v, causal=True, cp_world_size=2, cp_rank=0)
"
