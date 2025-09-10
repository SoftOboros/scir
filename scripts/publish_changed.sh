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
missing_on_registry=()

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

# Include crates that are not yet present on crates.io at the local version
exists_on_crates_io() {
  local name="$1"; local ver="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "https://crates.io/api/v1/crates/${name}/${ver}" >/dev/null 2>&1
    return $?
  fi
  # If curl is unavailable, err on the side of publishing
  return 1
}

for crate in "${crates[@]}"; do
  local_toml="crates/${crate}/Cargo.toml"
  [[ -f "$local_toml" ]] || continue
  v_local=$(get_version "$local_toml" || echo "")
  if [[ -n "$v_local" ]]; then
    if ! exists_on_crates_io "$crate" "$v_local"; then
      missing_on_registry+=("$crate")
    fi
  fi
done

# Merge and de-duplicate while keeping order defined in `crates` array
to_publish=()
for crate in "${crates[@]}"; do
  for c in "${changed[@]}"; do
    [[ "$crate" == "$c" ]] && to_publish+=("$crate") && break
  done
  for c in "${missing_on_registry[@]}"; do
    if [[ " ${to_publish[*]} " != *" $crate "* && "$crate" == "$c" ]]; then
      to_publish+=("$crate")
      break
    fi
  done
done

if [[ ${#to_publish[@]} -eq 0 ]]; then
  echo "No crate versions changed since $BASE and all present on crates.io. Nothing to publish."
  exit 0
fi

echo "Crates to publish (ordered): ${to_publish[*]}"

for crate in "${to_publish[@]}"; do
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
