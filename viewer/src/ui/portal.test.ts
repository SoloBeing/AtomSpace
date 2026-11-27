import { describe, expect, it } from 'vitest';
import type { Poi } from '../scene/loader';
import { portalInReach, portalUrl } from './portal';

const poi = (id: string, x: number, z: number, portal?: Poi['portal']): Poi => ({
  id,
  name: id,
  position: [x, 1.2, z],
  source: 'manual',
  ...(portal && { portal }),
});

describe('portals', () => {
  const door = poi('door', 0, 0, { scene: 'arkit-1', arrive: 'back-door' });
  const farDoor = poi('far-door', 5, 0, { scene: 'arkit-2' });
  const tv = poi('tv', 0.2, 0);

  it('offers the nearest doorway within reach, ignoring plain POIs', () => {
    expect(portalInReach([tv, door, farDoor], 0.5, 0.5)).toBe(door);
    expect(portalInReach([tv, door, farDoor], 2.5, 0)).toBeNull(); // 2.5 m from both
    expect(portalInReach([tv], 0.2, 0)).toBeNull();
  });

  it('links to the target scene and arrival point, keeping admin mode', () => {
    expect(portalUrl(door, new URLSearchParams('scene=clite'))).toBe('?scene=arkit-1&via=1&arrive=back-door');
    expect(portalUrl(farDoor, new URLSearchParams('scene=clite&admin'))).toBe('?scene=arkit-2&via=1&admin=');
    expect(new URLSearchParams(portalUrl(farDoor, new URLSearchParams('admin'))).has('admin')).toBe(true);
  });
});
