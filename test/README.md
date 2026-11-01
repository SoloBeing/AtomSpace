# Sample data

`testroom/frames/` holds 101 keyframes from a 90-second, 1080p phone video of our test room, chosen from 540 candidate frames by `atomspace ingest`. `scenes/testroom/ingest.json` lists them, so they go straight into reconstruction.

```bash
cp -r test/testroom/frames scenes/testroom/
atomspace reconstruct testroom
atomspace splat testroom
atomspace bundle testroom
```

Steps and viewer instructions are in the main [README](../README.md#reconstruct-the-sample-scene). You need a GPU machine for the first two commands: 8 GB of VRAM recommended, 4 GB minimum.

Reconstructed from these frames, expect a usable room with a black band at eye level, because the capture was tilted down throughout.

These frames aren't under the MIT License: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#images-and-sample-data).
