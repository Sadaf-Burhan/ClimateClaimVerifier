#!/usr/bin/env bash
# Rebuild every .d2 source in diagrams/ into a matching .svg.
#
#   ./build-diagrams.sh          # build all
#   ./build-diagrams.sh 06       # build only diagrams matching "06"
#
# Files starting with "_" are shared partials (imported via ...@_theme), not diagrams.
# 09 uses the elk layout — dagre routes its 8-way AND-gate fan-in badly.

set -euo pipefail
cd "$(dirname "$0")/diagrams"

D2_BIN="${D2_BIN:-d2}"
command -v "$D2_BIN" >/dev/null 2>&1 || D2_BIN="/c/Program Files/D2/d2.exe"
command -v "$D2_BIN" >/dev/null 2>&1 || {
  echo "d2 not found. Install: winget install Terrastruct.D2   (or set D2_BIN=/path/to/d2)" >&2
  exit 1
}

filter="${1:-}"
built=0

for src in *.d2; do
  [[ "$src" == _* ]] && continue
  [[ -n "$filter" && "$src" != *"$filter"* ]] && continue

  case "$src" in
    09-*) layout=elk ;;
    *)    layout=dagre ;;
  esac

  out="${src%.d2}.svg"
  "$D2_BIN" --layout="$layout" --theme=0 --pad=24 "$src" "$out"

  # D2's root <svg> carries only a viewBox — browsers can't size it inside an <img>
  # (height collapses in the scrollable panes). Inject width/height from the viewBox.
  python - "$out" <<'PY'
import re, sys
f = sys.argv[1]
s = open(f, encoding='utf-8').read()
m = re.search(r'<svg\b([^>]*?)viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
if m and ' width=' not in m.group(0):
    s = s[:m.start()] + f'<svg width="{m.group(2)}" height="{m.group(3)}"' + s[m.start()+4:]
    open(f, 'w', encoding='utf-8').write(s)
PY

  built=$((built + 1))
done

echo "built $built diagram(s)"
