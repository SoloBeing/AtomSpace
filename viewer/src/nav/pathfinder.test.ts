import { describe, expect, it } from 'vitest';
import type { Navmesh, Vec3 } from '../scene/loader';
import { Pathfinder, routeLength } from './pathfinder';

/** A navmesh of 1 m floor cells, two triangles each, neighbours filled in like the pipeline does. */
function gridNavmesh(cells: [number, number][], clockwise = false): Navmesh {
  const vertices: Vec3[] = [];
  const index = new Map<string, number>();
  const vertex = (x: number, z: number) => {
    const key = `${x},${z}`;
    if (!index.has(key)) {
      index.set(key, vertices.length);
      vertices.push([x, 0, z]);
    }
    return index.get(key)!;
  };
  const triangles: number[][] = [];
  for (const [x, z] of cells) {
    const [a, b, c, d] = [vertex(x, z), vertex(x + 1, z), vertex(x + 1, z + 1), vertex(x, z + 1)];
    triangles.push([a, c, b], [a, d, c]); // counter-clockwise seen from +Y
  }
  if (clockwise) triangles.forEach((t) => t.reverse());

  const owner = new Map<string, [number, number]>();
  const neighbours = triangles.map(() => [-1, -1, -1]);
  triangles.forEach((t, poly) => {
    for (let i = 0; i < 3; i++) {
      const key = [t[i], t[(i + 1) % 3]].sort((p, q) => p - q).join('-');
      const other = owner.get(key);
      if (other) {
        neighbours[poly][i] = other[0];
        neighbours[other[0]][other[1]] = poly;
      } else owner.set(key, [poly, i]);
    }
  });
  return {
    format_version: 1,
    units: 'm',
    agent: { radius_m: 0.25, height_m: 1.7, max_climb_m: 0.2 },
    vertices,
    polygons: triangles.map((v, i) => ({ v, n: neighbours[i] })),
  };
}

// An L: a corridor along x (z 0..1), turning up along z at x 2..3.
const L: [number, number][] = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]];

describe.each([false, true])('Pathfinder (clockwise triangles: %s)', (clockwise) => {
  const finder = new Pathfinder(gridNavmesh(L, clockwise));

  it('goes straight when nothing is in the way', () => {
    const path = finder.findPath([0.2, 0, 0.5], [2.8, 0, 0.5])!;
    expect(path).toEqual([
      [0.2, 0, 0.5],
      [2.8, 0, 0.5],
    ]);
  });

  it('bends exactly once, at the inside corner of the L', () => {
    const path = finder.findPath([0.5, 0, 0.5], [2.5, 0, 2.5])!;
    expect(path).toHaveLength(3);
    expect(path[1][0]).toBeCloseTo(2);
    expect(path[1][2]).toBeCloseTo(1);
    expect(routeLength(path)).toBeCloseTo(2 * Math.hypot(1.5, 0.5));
  });

  it('runs the same route backwards', () => {
    const there = finder.findPath([0.5, 0, 0.5], [2.5, 0, 2.5])!;
    const back = finder.findPath([2.5, 0, 2.5], [0.5, 0, 0.5])!;
    expect(routeLength(back)).toBeCloseTo(routeLength(there));
  });

  it('gives up on points off the navmesh', () => {
    expect(finder.findPath([0.5, 0, 0.5], [0.5, 0, 2.5])).toBeNull();
    expect(finder.findPath([-1, 0, 0.5], [2.5, 0, 2.5])).toBeNull();
  });
});

describe('Pathfinder around an obstacle', () => {
  // A 5 x 3 room with a 1 x 1 pillar in the middle cell (2, 1).
  const room: [number, number][] = [];
  for (let x = 0; x < 5; x++) for (let z = 0; z < 3; z++) if (!(x === 2 && z === 1)) room.push([x, z]);
  const finder = new Pathfinder(gridNavmesh(room));

  it('takes the short way round, touching the pillar at two corners', () => {
    const path = finder.findPath([0.5, 0, 1.4], [4.5, 0, 1.4])!;
    expect(path).toHaveLength(4);
    const corners = path.slice(1, 3).map(([x, , z]) => [x, z]);
    // Either side is equally short from z = 1.5; from 1.4 the lower side (z = 1) wins.
    expect(corners).toEqual([
      [2, 1],
      [3, 1],
    ]);
    expect(routeLength(path)).toBeCloseTo(2 * Math.hypot(1.5, 0.4) + 1);
  });

  it('reports a route between disconnected areas as missing', () => {
    const islands = new Pathfinder(gridNavmesh([[0, 0], [3, 0]]));
    expect(islands.findPath([0.5, 0, 0.5], [3.5, 0, 0.5])).toBeNull();
  });
});
