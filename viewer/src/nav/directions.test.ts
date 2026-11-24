import { describe, expect, it } from 'vitest';
import type { Navmesh, Vec3 } from '../scene/loader';
import { Directions, type DirectionsEvent } from './directions';
import { Pathfinder } from './pathfinder';
import type { RouteLine } from './routeLine';

// Two 2 x 2 m squares side by side (x 0..4, z 0..2), as four triangles.
const navmesh: Navmesh = {
  format_version: 1,
  units: 'm',
  agent: { radius_m: 0.25, height_m: 1.7, max_climb_m: 0.2 },
  vertices: [
    [0, 0, 0],
    [2, 0, 0],
    [4, 0, 0],
    [0, 0, 2],
    [2, 0, 2],
    [4, 0, 2],
  ],
  polygons: [
    { v: [0, 4, 1], n: [1, 2, -1] },
    { v: [0, 3, 4], n: [-1, -1, 0] },
    { v: [1, 4, 5], n: [0, -1, 3] }, // n[i] is across edge v[i] -> v[i + 1]
    { v: [1, 5, 2], n: [2, -1, -1] },
  ],
};

function recorder() {
  const shown: (Vec3[] | null)[] = [];
  const line: RouteLine = {
    show: (points) => shown.push(points),
    clear: () => shown.push(null),
  };
  return { line, shown };
}

describe('Directions', () => {
  const pathfinder = new Pathfinder(navmesh);

  it('shows the route without moving anyone', () => {
    const { line, shown } = recorder();
    const directions = new Directions(pathfinder, line);
    expect(directions.start([0.5, 0, 1], [3.5, 0, 1])).toBe(true);
    expect(shown.at(-1)![0]).toEqual([0.5, 0, 1]);
  });

  it('redraws from wherever the walker is now', () => {
    const { line, shown } = recorder();
    const directions = new Directions(pathfinder, line);
    directions.start([0.5, 0, 1], [3.5, 0, 1]);
    directions.update([2, 0, 1]);
    expect(shown.at(-1)![0]).toEqual([2, 0, 1]);
    expect(shown.at(-1)!.at(-1)).toEqual([3.5, 0, 1]);
  });

  it('clears the route on arrival', () => {
    const { line, shown } = recorder();
    const directions = new Directions(pathfinder, line);
    directions.start([0.5, 0, 1], [3.5, 0, 1]);
    directions.update([3.2, 0, 1.1]);
    expect(shown.at(-1)).toBeNull();
  });

  it('shows nothing when there is no route', () => {
    const { line, shown } = recorder();
    const directions = new Directions(pathfinder, line);
    expect(directions.start([0.5, 0, 1], [9, 0, 9])).toBe(false);
    expect(shown.at(-1)).toBeNull();
  });

  it('reports the distance left as the walker goes, then the arrival', () => {
    const events: DirectionsEvent[] = [];
    const directions = new Directions(pathfinder, recorder().line, (e) => events.push(e));
    directions.start([0.5, 0, 1], [3.5, 0, 1]);
    directions.update([2.5, 0, 1]);
    directions.update([3.2, 0, 1]);
    expect(events).toEqual([
      { kind: 'route', metres: 3 },
      { kind: 'route', metres: 1 },
      { kind: 'arrived' },
    ]);
  });

  it('reports a stop, not an arrival, when there is no route', () => {
    const events: DirectionsEvent[] = [];
    const directions = new Directions(pathfinder, recorder().line, (e) => events.push(e));
    directions.start([0.5, 0, 1], [9, 0, 9]);
    expect(events).toEqual([{ kind: 'stopped' }]);
  });

  it('keeps the last route if one step lands just off the mesh', () => {
    const { line, shown } = recorder();
    const directions = new Directions(pathfinder, line);
    directions.start([0.5, 0, 1], [3.5, 0, 1]);
    const before = shown.length;
    directions.update([-0.01, 0, 1]);
    expect(shown.length).toBe(before);
  });
});
