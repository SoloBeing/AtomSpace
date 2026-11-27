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

## Synthetic sensor fixture

`strayroom/` is a generated Stray Scanner recording of a plain 4 x 3 x 2.6 m box room: twelve
poses on a loop, depth and confidence PNGs, and pre-extracted frames. No video, so none of it
goes to Git LFS. Regenerate it with:

```bash
python pipeline/scripts/make_stray_fixture.py --out test/strayroom
```

It exists to pin the arithmetic of the sensor track — millimetre depth at a seventh of the
colour resolution, intrinsics quoted for the colour frame, ARKit's pose convention — because
getting any of those wrong yields a cloud that still looks like a room and is quietly the
wrong size. Being consistent by construction, it cannot stand in for a real capture: index
correspondence, the container rotation flag, and whether LiDAR plus MASt3R actually beats
vision alone all need a recording from a real device. Unlike `testroom/`, this fixture is
ours and is covered by the MIT License.
