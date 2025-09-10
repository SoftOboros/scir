#!/usr/bin/env bash
set -euo pipefail

# Base ref to compare versions against (default to origin/main or previous SHA)
BASE=${1:-origin/main}

# Ordered publish list to respect dependencies
crates=(
  scir-core
  scir-nd
  scir-fft
  scir-signal
  scir-optimize
  scir-gpu
  scir # umbrella last
)

changed=()

get_version() {
  awk -F '"' '/^version\s*=/{print $2; exit}' "$1"
}

for crate in "${crates[@]}"; do
  local_toml="crates/${crate}/Cargo.toml"
  if [[ ! -f "$local_toml" ]]; then
    continue
  fi
  v_local=$(get_version "$local_toml" || echo "")
  v_base=$(git show "$BASE:$local_toml" 2>/dev/null | awk -F '"' '/^version\s*=/{print $2; exit}' || echo "")
  if [[ -n "$v_local" && "$v_local" != "$v_base" ]]; then
    changed+=("$crate")
  fi
done

if [[ ${#changed[@]} -eq 0 ]]; then
  echo "No crate versions changed since $BASE. Nothing to publish."
  exit 0
fi

echo "Crates to publish (ordered): ${changed[*]}"

for crate in "${changed[@]}"; do
  echo "::group::Publish $crate"
  # Skip crates explicitly marked as non-publishable
  if grep -qE '^publish\s*=\s*false' "crates/${crate}/Cargo.toml" 2>/dev/null; then
    echo "Skipping $crate (publish = false)"
    echo "::endgroup::"
    continue
  fi
  # Try publish; allow failures to be bypassed with a message
  if ! cargo publish -p "$crate" --token "${CARGO_REGISTRY_TOKEN:-}" --no-verify; then
    echo "⚠️ publish $crate failed (likely due to path dependencies or prior version). Skipping."
  fi
  echo "::endgroup::"
done
