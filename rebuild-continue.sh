#!/bin/bash
# Rebuild Continue.dev from source fork and install into VS Code
# Run this after making changes to the Continue.dev fork

set -e

CONTINUE_DIR="$HOME/repos/continue_dev"
VSCODE_EXT_DIR="$CONTINUE_DIR/extensions/vscode"

echo "Building Continue.dev from source..."

# Build internal packages
for pkg in config-types llm-info fetch terminal-security openai-adapters config-yaml; do
    echo "  Building @continuedev/$pkg..."
    cd "$CONTINUE_DIR/packages/$pkg"
    npm run build --silent 2>/dev/null
done

# Build GUI
echo "  Building GUI..."
cd "$CONTINUE_DIR/gui"
npm run build --silent 2>&1 | tail -1

# Package extension
echo "  Packaging VSIX..."
cd "$VSCODE_EXT_DIR"
npm run package --silent 2>&1 | tail -1

# Find the built VSIX
VSIX=$(ls -t "$VSCODE_EXT_DIR/build/"*.vsix 2>/dev/null | head -1)

if [ -z "$VSIX" ]; then
    echo "ERROR: No VSIX file found in $VSCODE_EXT_DIR/build/"
    exit 1
fi

echo "  Installing $VSIX..."
/usr/local/bin/code --install-extension "$VSIX" --force 2>&1 | tail -1

echo ""
echo "Done. Reload VS Code window to pick up changes."
