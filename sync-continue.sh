#!/bin/bash
# Sync Continue.dev fork with upstream updates
#
# Strategy:
#   1. Fetch latest from upstream (continuedev/continue)
#   2. Update local main to match upstream/main
#   3. Rebase our custom-supercoder branch on top
#   4. Rebuild the VSIX
#
# Our changes are isolated to ~20 GUI component files (nulled out UI for
# sign-in, onboarding, indexing, organizations). Conflicts should be rare
# and easy to resolve since we replaced entire file contents with stubs.
#
# Usage: ./sync-continue.sh [--rebuild]
#   --rebuild   Also rebuild and install the VSIX after syncing

set -euo pipefail

CONTINUE_DIR="$HOME/repos/continue_dev"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$CONTINUE_DIR"

echo "=== Syncing Continue.dev fork with upstream ==="

# Ensure upstream remote exists
if ! git remote | grep -q upstream; then
    echo "Adding upstream remote..."
    git remote add upstream https://github.com/continuedev/continue.git
fi

echo ""
echo "1. Fetching upstream..."
git fetch upstream

echo ""
echo "2. Updating local main..."
git checkout main
git merge upstream/main --ff-only || {
    echo "ERROR: main has diverged from upstream. Manual merge needed."
    echo "  Run: git merge upstream/main"
    exit 1
}

echo ""
echo "3. Pushing updated main to origin..."
git push origin main

echo ""
echo "4. Rebasing custom-supercoder on updated main..."
git checkout custom-supercoder
git rebase main || {
    echo ""
    echo "REBASE CONFLICT detected."
    echo "Our customizations touch these files (all are stub replacements):"
    echo ""
    git diff --name-only main...custom-supercoder 2>/dev/null || true
    echo ""
    echo "To resolve:"
    echo "  1. Fix conflicts in the listed files"
    echo "  2. git add <resolved files>"
    echo "  3. git rebase --continue"
    echo "  4. Re-run: $0 --rebuild"
    exit 1
}

echo ""
echo "5. Pushing rebased branch to origin..."
git push origin custom-supercoder --force-with-lease

echo ""
echo "Sync complete."

if [[ "${1:-}" == "--rebuild" ]]; then
    echo ""
    echo "6. Rebuilding VSIX..."
    "$SCRIPT_DIR/rebuild-continue.sh"
fi
