import * as THREE from 'three';
import type { Navmesh, Vec3 } from '../scene/loader';

export const EYE_HEIGHT_M = 1.6;

/** The floor point under an eye-height camera position: routes and the navmesh work on the floor. */
export const toFloor = (eye: THREE.Vector3): Vec3 => [eye.x, eye.y - EYE_HEIGHT_M, eye.z];

const EDGE_TOLERANCE = 1e-9;

type Triangle = [Vec3, Vec3, Vec3];

export class NavmeshGuard {
  private readonly triangles: Triangle[];
  /** Outline edges of the walkable area (used by exactly one polygon), as [x0, z0, x1, z1]. */
  private readonly boundary: [number, number, number, number][];

  constructor(navmesh: Navmesh) {
    // Polygons are convex, so a fan covers them exactly.
    this.triangles = navmesh.polygons.flatMap(({ v }) =>
      v.slice(1, -1).map((_, i): Triangle => [navmesh.vertices[v[0]], navmesh.vertices[v[i + 1]], navmesh.vertices[v[i + 2]]]),
    );
    const uses = new Map<string, number>();
    const key = (a: number, b: number) => (a < b ? `${a},${b}` : `${b},${a}`);
    for (const { v } of navmesh.polygons) {
      for (let i = 0; i < v.length; i++) {
        const k = key(v[i], v[(i + 1) % v.length]);
        uses.set(k, (uses.get(k) ?? 0) + 1);
      }
    }
    this.boundary = [...uses].filter(([, n]) => n === 1).map(([k]) => {
      const [a, b] = k.split(',').map((i) => navmesh.vertices[Number(i)]);
      return [a[0], a[2], b[0], b[2]];
    });
  }

  /** Floor height at (x, z), or null if that point is outside the walkable navmesh. */
  floorHeightAt(x: number, z: number): number | null {
    for (const [a, b, c] of this.triangles) {
      const det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
      if (det === 0) continue;
      const u = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det;
      const v = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det;
      const w = 1 - u - v;
      if (u >= -EDGE_TOLERANCE && v >= -EDGE_TOLERANCE && w >= -EDGE_TOLERANCE) {
        return u * a[1] + v * b[1] + w * c[1];
      }
    }
    return null;
  }

  /**
   * Lookahead check: a step that would leave the navmesh slides along the edge it hits instead
   * (only the part of the step running along that edge), and stops dead if even that leaves.
   */
  tryMove(position: THREE.Vector3, delta: THREE.Vector3): THREE.Vector3 {
    const onMesh = (step: THREE.Vector3): THREE.Vector3 | null => {
      const next = position.clone().add(step);
      const floorY = this.floorHeightAt(next.x, next.z);
      return floorY === null ? null : next.setY(floorY + EYE_HEIGHT_M);
    };
    const direct = onMesh(delta);
    if (direct) return direct;
    const edge = this.nearestBoundaryEdge(position.x + delta.x, position.z + delta.z);
    if (!edge) return position.clone();
    const along = new THREE.Vector3(edge[2] - edge[0], 0, edge[3] - edge[1]).normalize();
    return onMesh(along.multiplyScalar(along.dot(delta))) ?? position.clone();
  }

  private nearestBoundaryEdge(x: number, z: number): [number, number, number, number] | null {
    let best: [number, number, number, number] | null = null;
    let bestDistance = Infinity;
    for (const e of this.boundary) {
      const [x0, z0, x1, z1] = e;
      const dx = x1 - x0;
      const dz = z1 - z0;
      const t = THREE.MathUtils.clamp(((x - x0) * dx + (z - z0) * dz) / (dx * dx + dz * dz || 1), 0, 1);
      const distance = (x0 + t * dx - x) ** 2 + (z0 + t * dz - z) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = e;
      }
    }
    return best;
  }

  /** The closest eye-height camera position on the navmesh to (x, z). */
  nearestPosition(x: number, z: number): THREE.Vector3 {
    const target = new THREE.Vector3(x, 0, z);
    const closest = new THREE.Vector3();
    const best = new THREE.Vector3();
    let bestDistance = Infinity;
    const triangle = new THREE.Triangle();
    for (const [a, b, c] of this.triangles) {
      target.y = (a[1] + b[1] + c[1]) / 3;
      triangle.a.fromArray(a);
      triangle.b.fromArray(b);
      triangle.c.fromArray(c);
      triangle.closestPointToPoint(target, closest);
      const distance = closest.distanceToSquared(target);
      if (distance < bestDistance) {
        bestDistance = distance;
        best.copy(closest);
      }
    }
    return best.setY(best.y + EYE_HEIGHT_M);
  }

  /** A starting camera position: the middle of the largest polygon, well away from any edge. */
  spawnPosition(): THREE.Vector3 {
    const area = ([a, b, c]: Triangle) =>
      Math.abs((b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]));
    const largest = this.triangles.reduce((best, t) => (area(t) > area(best) ? t : best));
    const center = (i: number) => (largest[0][i] + largest[1][i] + largest[2][i]) / 3;
    return new THREE.Vector3(center(0), center(1) + EYE_HEIGHT_M, center(2));
  }
}
