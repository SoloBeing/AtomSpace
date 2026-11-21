import type { Vec3 } from '../scene/loader';
import { type Pathfinder, routeLength } from './pathfinder';
import type { RouteLine } from './routeLine';

export const ARRIVED_M = 0.5;

/** What the walker should be told: how far is left, that they've arrived, or that there's no route. */
export type DirectionsEvent = { kind: 'route'; metres: number } | { kind: 'arrived' } | { kind: 'stopped' };

/**
 * Turn-by-turn style directions: shows the route to a destination and keeps it starting at the
 * walker's feet as they move, until they arrive. Positions are on the floor, not at eye height.
 */
export class Directions {
  private goal: Vec3 | null = null;

  constructor(
    private readonly pathfinder: Pathfinder,
    private readonly line: RouteLine,
    private readonly onChange: (event: DirectionsEvent) => void = () => {},
  ) {}

  get active(): boolean {
    return this.goal !== null;
  }

  /** Shows the route from `from` to `goal`. Returns false, showing nothing, if there is none. */
  start(from: Vec3, goal: Vec3): boolean {
    const path = this.pathfinder.findPath(from, goal);
    if (!path) {
      this.stop();
      return false;
    }
    this.goal = goal;
    this.show(path);
    return true;
  }

  /** Call when the walker has moved: redraws from their new spot, or clears on arrival. */
  update(from: Vec3): void {
    if (!this.goal) return;
    if (Math.hypot(this.goal[0] - from[0], this.goal[2] - from[2]) < ARRIVED_M) {
      this.clear();
      this.onChange({ kind: 'arrived' });
      return;
    }
    const path = this.pathfinder.findPath(from, this.goal);
    if (path) this.show(path); // else keep the last route: a step along an edge can briefly miss the mesh
  }

  stop(): void {
    this.clear();
    this.onChange({ kind: 'stopped' });
  }

  private show(path: Vec3[]): void {
    this.line.show(path);
    this.onChange({ kind: 'route', metres: routeLength(path) });
  }

  private clear(): void {
    this.goal = null;
    this.line.clear();
  }
}
