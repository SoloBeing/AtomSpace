import * as THREE from 'three';
import type { NavmeshGuard } from '../controls/navmeshGuard';
import { EYE_HEIGHT_M } from '../controls/navmeshGuard';
import type { Pathfinder } from '../nav/pathfinder';
import type { RouteLine } from '../nav/routeLine';
import type { Poi, Vec3 } from '../scene/loader';

const STANDOFF_M = 1.5;
const GUIDE_SPEED_MPS = 2.5; // brisker than walking, slow enough to follow the turns
const MIN_WALK_MS = 800;
const MAX_WALK_MS = 8000;
const FINAL_TURN_MS = 700;
const LOOK_AHEAD_M = 0.8; // face a point this far along the route, so corners are turned smoothly
const TURN_RATE = 6; // per second: how quickly the view catches up with the route's direction

let currentFlight = 0;

/** Where the camera lands: poi.view, or a point STANDOFF_M from the POI on the side we're coming from. */
function landing(camera: THREE.PerspectiveCamera, poi: Poi, guard: NavmeshGuard | null) {
  const target = new THREE.Vector3().fromArray(poi.view?.target ?? poi.position);
  let position: THREE.Vector3;
  if (poi.view) {
    position = new THREE.Vector3().fromArray(poi.view.position);
  } else {
    const away = camera.position.clone().sub(target).setY(0);
    if (away.lengthSq() === 0) away.set(0, 0, 1);
    position = target.clone().addScaledVector(away.normalize(), STANDOFF_M).setY(EYE_HEIGHT_M);
  }
  return { position: guard ? guard.nearestPosition(position.x, position.z) : position, target };
}

const lookRotation = (from: THREE.Vector3, to: THREE.Vector3, up: THREE.Vector3) =>
  new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(from, to, up));

/**
 * Smoothly animates the camera to poi.view (or a standoff point in front of poi.position),
 * ending on the navmesh so walking can resume from there. Starting another flight cancels this one.
 */
export function flyTo(
  camera: THREE.PerspectiveCamera,
  poi: Poi,
  guard: NavmeshGuard | null,
  durationMs = 1200,
): Promise<void> {
  const flight = ++currentFlight;
  const { position, target } = landing(camera, poi, guard);
  const startPosition = camera.position.clone();
  const startRotation = camera.quaternion.clone();
  const endRotation = lookRotation(position, target, camera.up);
  const start = performance.now();

  return new Promise((resolve) => {
    const step = (now: number) => {
      if (flight !== currentFlight) return resolve();
      const t = Math.min((now - start) / durationMs, 1);
      const eased = THREE.MathUtils.smootherstep(t, 0, 1);
      camera.position.lerpVectors(startPosition, position, eased);
      camera.quaternion.slerpQuaternions(startRotation, endRotation, eased);
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
}

/** The point `distance` meters along a polyline, clamped to its ends. */
export function pointAlong(points: THREE.Vector3[], distance: number): THREE.Vector3 {
  let left = Math.max(distance, 0);
  for (let i = 1; i < points.length; i++) {
    const step = points[i].distanceTo(points[i - 1]);
    if (left <= step) return points[i - 1].clone().lerp(points[i], step > 0 ? left / step : 1);
    left -= step;
  }
  return points[points.length - 1].clone();
}

/**
 * Walks the camera to the POI along the shortest route over the navmesh, drawing the route on the
 * floor, then turns to face the POI. Falls back to flyTo when there is no route.
 * Starting another flight cancels this one; the drawn route stays until the caller clears it.
 */
export function walkTo(
  camera: THREE.PerspectiveCamera,
  poi: Poi,
  guard: NavmeshGuard,
  pathfinder: Pathfinder,
  route: RouteLine,
): Promise<void> {
  const { position, target } = landing(camera, poi, guard);
  const feet = (p: THREE.Vector3): Vec3 => [p.x, p.y - EYE_HEIGHT_M, p.z];
  const path = pathfinder.findPath(feet(camera.position), feet(position));
  if (!path) {
    route.clear();
    return flyTo(camera, poi, guard);
  }

  const flight = ++currentFlight;
  route.show(path);
  const points = path.map(([x, y, z]) => new THREE.Vector3(x, y + EYE_HEIGHT_M, z));
  let total = 0;
  for (let i = 1; i < points.length; i++) total += points[i].distanceTo(points[i - 1]);
  const walkMs = total < 0.05 ? 0 : THREE.MathUtils.clamp((total / GUIDE_SPEED_MPS) * 1000, MIN_WALK_MS, MAX_WALK_MS);
  const endRotation = lookRotation(position, target, camera.up);
  const start = performance.now();
  let last = start;
  let turnFrom: THREE.Quaternion | null = null;

  return new Promise((resolve) => {
    const step = (now: number) => {
      if (flight !== currentFlight) return resolve();
      const dt = (now - last) / 1000;
      last = now;
      const elapsed = now - start;

      if (elapsed < walkMs) {
        const travelled = THREE.MathUtils.smootherstep(elapsed / walkMs, 0, 1) * total;
        camera.position.copy(pointAlong(points, travelled));
        const ahead = pointAlong(points, travelled + LOOK_AHEAD_M).setY(camera.position.y);
        if (ahead.distanceToSquared(camera.position) > 0.01) {
          const heading = lookRotation(camera.position, ahead, camera.up);
          camera.quaternion.slerp(heading, 1 - Math.exp(-TURN_RATE * dt));
        }
        requestAnimationFrame(step);
        return;
      }

      // Arrived: settle on the landing point and turn to face the POI.
      camera.position.copy(position);
      turnFrom ??= camera.quaternion.clone();
      const t = Math.min((elapsed - walkMs) / FINAL_TURN_MS, 1);
      camera.quaternion.slerpQuaternions(turnFrom, endRotation, THREE.MathUtils.smootherstep(t, 0, 1));
      if (t < 1) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
}
