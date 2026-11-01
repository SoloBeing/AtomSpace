#!/usr/bin/env bash
# Downloads the MASt3R checkpoints (CC BY-NC-SA 4.0, non-commercial) from Hugging Face into
# checkpoints/. Each file goes to <name>.part, is verified, then renamed, so a final path only
# ever holds a complete, verified file. Safe to interrupt and re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/checkpoints"
MODEL="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

# Pinned revisions, so nobody's later upload can change what we download.
HF="https://huggingface.co"
OFFICIAL="$HF/naver/$MODEL/resolve/06e7259f34c3060f322df5cb0c7b9094f57e41fc"
# Naver doesn't publish the retrieval files on Hugging Face. These are community copies; their
# hashes agree with each other and with Naver's own server.
RETRIEVAL="$HF/pablovela5620/mast3r-slam/resolve/c6842fe1801f35e802a640c5bff73a19d54ca3cb"
RETRIEVAL_BACKUP="$HF/vslamlab/mast3rslam_weights/resolve/5eec4450715c77a4f2a823750ff6a64feae2a717"

# path under checkpoints/|size in bytes|sha256|name at the source|sources, space separated
WEIGHTS=(
  "$MODEL/config.json|546|718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4|config.json|$OFFICIAL"
  "$MODEL/model.safetensors|2754661648|0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb|model.safetensors|$OFFICIAL"
  "${MODEL}_retrieval_trainingfree.pth|8399111|000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50|${MODEL}_retrieval_trainingfree.pth|$RETRIEVAL $RETRIEVAL_BACKUP"
  "${MODEL}_retrieval_codebook.pkl|268435691|700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a|${MODEL}_retrieval_codebook.pkl|$RETRIEVAL $RETRIEVAL_BACKUP"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for tool in curl sha256sum; do
  command -v "$tool" >/dev/null || die "Missing $tool. Run: sudo apt install -y curl coreutils"
done

# verify <file> <size> <sha256>
verify() {
  local actual
  actual="$(stat -c %s "$1")"
  if [[ "$actual" != "$2" ]]; then
    echo "    size mismatch: $actual bytes, expected $2" >&2
    return 1
  fi
  echo "    computing sha256..."
  actual="$(sha256sum "$1" | cut -d' ' -f1)"
  if [[ "$actual" != "$3" ]]; then
    echo "    sha256 mismatch: $actual, expected $3" >&2
    return 1
  fi
}

for entry in "${WEIGHTS[@]}"; do
  IFS='|' read -r path size sha name sources <<<"$entry"
  final="$DEST/$path"
  part="$final.part"
  mkdir -p "$(dirname "$final")"
  echo "==> $path"

  if [[ -f "$final" ]]; then
    if verify "$final" "$size" "$sha"; then
      echo "    OK (already downloaded)"
      continue
    fi
    echo "    existing file is bad; deleting and downloading again"
    rm -f "$final"
  fi

  downloaded=false
  for base in $sources; do
    if [[ ! -f "$part" || "$(stat -c %s "$part")" -lt "$size" ]]; then
      echo "    from $base"
      curl -fL --retry 3 --retry-delay 5 -C - -o "$part" "$base/$name" || continue
    fi
    if verify "$part" "$size" "$sha"; then
      downloaded=true
      break
    fi
    rm -f "$part"
  done

  [[ "$downloaded" == true ]] || die "Could not get a verified copy of $path. Re-run to try again."
  mv -f "$part" "$final"
  echo "    OK"
done
echo "All weights verified in $DEST"
