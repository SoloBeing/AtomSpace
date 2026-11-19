import type { Navmesh, Vec3 } from '../scene/loader';

type Point = [number, number]; // (x, z) on the floor

const EPSILON = 1e-9;

/** Twice the signed area of a, b, c on the floor; the sign says which side of a→b c lies on. */
function triarea2(a: Point, b: Point, c: Point): number {
  return (c[0] - a[0]) * (b[1] - a[1]) - (b[0] - a[0]) * (c[1] - a[1]);
}

const same = (a: Point, b: Point) => Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON;
const distance = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

interface Polygon {
  corners: Point[];
  heights: number[];
  neighbours: number[]; // neighbours[i] is across the edge corners[i] → corners[i + 1]
}

/** Binary heap of polygon indices ordered by score; stale entries are skipped by the caller. */
class Queue {
  private items: { index: number; score: number }[] = [];
  get size() {
    return this.items.length;
  }
  push(index: number, score: number) {
    const items = this.items;
    items.push({ index, score });
    for (let i = items.length - 1; i > 0; ) {
      const parent = (i - 1) >> 1;
      if (items[parent].score <= items[i].score) break;
      [items[parent], items[i]] = [items[i], items[parent]];
      i = parent;
    }
  }
  pop(): number {
    const items = this.items;
    const top = items[0].index;
    const last = items.pop()!;
    if (items.length > 0) {
      items[0] = last;
      for (let i = 0; ; ) {
        const l = 2 * i + 1;
        const r = l + 1;
        let smallest = i;
        if (l < items.length && items[l].score < items[smallest].score) smallest = l;
        if (r < items.length && items[r].score < items[smallest].score) smallest = r;
        if (smallest === i) break;
        [items[smallest], items[i]] = [items[i], items[smallest]];
        i = smallest;
      }
    }
    return top;
  }
}

/**
 * Shortest walking routes over the navmesh: A* across polygons, then the funnel algorithm pulls
 * the route tight so it runs straight between the corners it has to go around.
 */
export class Pathfinder {
  private readonly polygons: Polygon[];

  constructor(navmesh: Navmesh) {
    this.polygons = navmesh.polygons.map(({ v, n }) => ({
      corners: v.map((i): Point => [navmesh.vertices[i][0], navmesh.vertices[i][2]]),
      heights: v.map((i) => navmesh.vertices[i][1]),
      neighbours: v.map((_, i) => n?.[i] ?? -1),
    }));
  }

  /** Index of the polygon under (x, z), or -1 if that point is off the navmesh. */
  locate(x: number, z: number): number {
    const p: Point = [x, z];
    return this.polygons.findIndex(({ corners }) => {
      // Inside a convex polygon means on the same side of every edge (either winding).
      let sign = 0;
      for (let i = 0; i < corners.length; i++) {
        const s = triarea2(corners[i], corners[(i + 1) % corners.length], p);
        if (Math.abs(s) < EPSILON) continue;
        if (sign === 0) sign = Math.sign(s);
        else if (Math.sign(s) !== sign) return false;
      }
      return true;
    });
  }

  /** Floor height inside polygon `index` at (x, z), from its first triangle's plane. */
  private heightAt(index: number, [x, z]: Point): number {
    const { corners: [a, b, c], heights: [ha, hb, hc] } = this.polygons[index];
    const det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if (Math.abs(det) < EPSILON) return ha;
    const u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (z - c[1])) / det;
    const v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (z - c[1])) / det;
    return u * ha + v * hb + (1 - u - v) * hc;
  }

  /**
   * Floor points from start to goal, both included, or null if either is off the navmesh or
   * they aren't connected. Heights follow the navmesh.
   */
  findPath(start: Vec3, goal: Vec3): Vec3[] | null {
    const from: Point = [start[0], start[2]];
    const to: Point = [goal[0], goal[2]];
    const startPoly = this.locate(...from);
    const goalPoly = this.locate(...to);
    if (startPoly === -1 || goalPoly === -1) return null;

    const corridor = this.searchPolygons(startPoly, goalPoly, from, to);
    if (!corridor) return null;
    const portals = this.portals(corridor, from, to);
    const points = stringPull(portals);

    // Heights: each corner lies on a portal or an endpoint; look up whichever polygon holds it.
    return points.map((p, i): Vec3 => {
      if (i === 0) return [p[0], this.heightAt(startPoly, p), p[1]];
      if (i === points.length - 1) return [p[0], this.heightAt(goalPoly, p), p[1]];
      const holder = corridor.find((index) => this.polygons[index].corners.some((c) => same(c, p))) ?? goalPoly;
      return [p[0], this.heightAt(holder, p), p[1]];
    });
  }

  /** A* over polygons; a polygon's position is where the route entered it (the portal midpoint). */
  private searchPolygons(startPoly: number, goalPoly: number, from: Point, to: Point): number[] | null {
    const cost = new Map<number, number>([[startPoly, 0]]);
    const position = new Map<number, Point>([[startPoly, from]]);
    const parent = new Map<number, number>();
    const closed = new Set<number>();
    const open = new Queue();
    open.push(startPoly, distance(from, to));

    while (open.size > 0) {
      const current = open.pop();
      if (closed.has(current)) continue;
      if (current === goalPoly) break;
      closed.add(current);
      const here = position.get(current)!;
      const { corners, neighbours } = this.polygons[current];
      neighbours.forEach((next, i) => {
        if (next === -1 || closed.has(next)) return;
        const a = corners[i];
        const b = corners[(i + 1) % corners.length];
        const entry: Point = next === goalPoly ? to : [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const g = cost.get(current)! + distance(here, entry);
        if (g >= (cost.get(next) ?? Infinity)) return;
        cost.set(next, g);
        position.set(next, entry);
        parent.set(next, current);
        open.push(next, g + distance(entry, to));
      });
    }

    if (startPoly !== goalPoly && !parent.has(goalPoly)) return null;
    const corridor = [goalPoly];
    while (corridor[0] !== startPoly) corridor.unshift(parent.get(corridor[0])!);
    return corridor;
  }

  /** The shared edges along the corridor as [left, right] pairs, framed by the start and goal. */
  private portals(corridor: number[], from: Point, to: Point): [Point, Point][] {
    const portals: [Point, Point][] = [[from, from]];
    for (let k = 0; k < corridor.length - 1; k++) {
      const { corners, neighbours } = this.polygons[corridor[k]];
      const i = neighbours.indexOf(corridor[k + 1]);
      const a = corners[i];
      const b = corners[(i + 1) % corners.length];
      // Which end is on the left depends on the polygon's winding; check against its middle.
      const middle: Point = [
        corners.reduce((s, c) => s + c[0], 0) / corners.length,
        corners.reduce((s, c) => s + c[1], 0) / corners.length,
      ];
      portals.push(triarea2(middle, a, b) > 0 ? [a, b] : [b, a]);
    }
    portals.push([to, to]);
    return portals;
  }
}

/** The "simple stupid funnel algorithm" (Mononen): the taut string through a list of portals. */
function stringPull(portals: [Point, Point][]): Point[] {
  const points: Point[] = [portals[0][0]];
  const add = (p: Point) => {
    if (!same(points[points.length - 1], p)) points.push(p);
  };
  let apex = portals[0][0];
  let left = portals[0][0];
  let right = portals[0][1];
  let apexIndex = 0;
  let leftIndex = 0;
  let rightIndex = 0;

  for (let i = 1; i < portals.length; i++) {
    const [newLeft, newRight] = portals[i];

    // Narrow the funnel from the right.
    if (triarea2(apex, right, newRight) <= 0) {
      if (same(apex, right) || triarea2(apex, left, newRight) > 0) {
        right = newRight;
        rightIndex = i;
      } else {
        // The right side crossed the left: the left point is a corner of the route.
        add(left);
        apex = right = left;
        apexIndex = rightIndex = leftIndex;
        i = apexIndex;
        continue;
      }
    }

    // Narrow the funnel from the left.
    if (triarea2(apex, left, newLeft) >= 0) {
      if (same(apex, left) || triarea2(apex, right, newLeft) < 0) {
        left = newLeft;
        leftIndex = i;
      } else {
        add(right);
        apex = left = right;
        apexIndex = leftIndex = rightIndex;
        i = apexIndex;
        continue;
      }
    }
  }

  add(portals[portals.length - 1][0]);
  return points;
}

/** Length of a route in meters. */
export function routeLength(points: Vec3[]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1], points[i][2] - points[i - 1][2]);
  }
  return total;
}
