import { describe, expect, it } from 'vitest';
import type { Navmesh } from '../scene/loader';
import { navmeshLines } from './navmeshOverlay';

// A 2 x 1 m rectangle split into two triangles along the diagonal (0,0)-(2,1).
const navmesh: Navmesh = {
  format_version: 1,
  units: 'm',
  agent: { radius_m: 0.25, height_m: 1.7, max_climb_m: 0.2 },
  vertices: [
    [0, 0, 0],
    [2, 0, 0],
    [2, 0, 1],
    [0, 0, 1],
  ],
  polygons: [
    { v: [0, 2, 1], n: [1, -1, -1] },
    { v: [0, 3, 2], n: [-1, -1, 0] },
  ],
};

describe('navmeshLines', () => {
  const lines = navmeshLines(navmesh);

  it('fills every triangle and measures the area', () => {
    expect(lines.fill.length).toBe(2 * 9);
    expect(lines.areaM2).toBeCloseTo(2);
  });

  it('draws the shared diagonal once and the four sides as the outline', () => {
    expect(lines.inner.length).toBe(6);
    expect(lines.outline.length).toBe(4 * 6);
  });

  it('lifts the lines just above the floor', () => {
    const heights = [...lines.fill].filter((_, i) => i % 3 === 1);
    expect(heights.every((y) => y > 0 && y < 0.05)).toBe(true);
  });
});
