#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle="$script_dir/build/Hibiki Test.app"
contents="$bundle/Contents"

swift build --package-path "$script_dir" -c release

if [[ -e "$bundle" ]]; then
  find "$bundle" -depth -delete
fi
mkdir -p "$contents/MacOS"
cp "$script_dir/.build/release/HibikiTestApp" "$contents/MacOS/HibikiTestApp"
cp "$script_dir/Info.plist" "$contents/Info.plist"
codesign --force --sign - --timestamp=none "$bundle"

echo "$bundle"
