#!/usr/bin/env bash
# Compiles the app's platform-free Kotlin (object capture + YOLO decoder) with kotlinc and runs
# the checks in Main.kt. Usage: ./run.sh <fixture_out_dir>   (needs kotlinc on PATH or $KOTLINC)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/../../android/app/src/main/java/com/cleanscan/app"
KOTLINC="${KOTLINC:-kotlinc}"
BUILD="$(mktemp -d)"
"$KOTLINC" -nowarn "$SRC/ObjectCapture.kt" "$SRC/YoloSegDecoder.kt" "$SRC/DepthProtocol.kt" "$HERE/Main.kt" -include-runtime -d "$BUILD/app-logic.jar"
java -jar "$BUILD/app-logic.jar" "${1:-$BUILD/fixture}"
