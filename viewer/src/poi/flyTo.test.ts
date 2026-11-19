import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { pointAlong } from './flyTo';

describe('pointAlong', () => {
  // 3 m along x, then 4 m along z.
  const route = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(3, 0, 0), new THREE.Vector3(3, 0, 4)];

  it('finds points on each leg', () => {
    expect(pointAlong(route, 1.5).toArray()).toEqual([1.5, 0, 0]);
    expect(pointAlong(route, 5).toArray()).toEqual([3, 0, 2]);
  });

  it('stops at the ends', () => {
    expect(pointAlong(route, -1).toArray()).toEqual([0, 0, 0]);
    expect(pointAlong(route, 100).toArray()).toEqual([3, 0, 4]);
  });

  it('copes with a repeated point', () => {
    const repeated = [route[0], route[0], route[1]];
    expect(pointAlong(repeated, 1).toArray()).toEqual([1, 0, 0]);
  });
});
