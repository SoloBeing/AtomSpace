import * as THREE from 'three';
import type { NavmeshGuard } from '../controls/navmeshGuard';
import { EYE_HEIGHT_M } from '../controls/navmeshGuard';
import type { Poi } from '../scene/loader';

const STANDOFF_M = 1.5;

let currentFlight = 0;

/**
 * Where to stand to see a POI: poi.view, or a point STANDOFF_M from the POI on the side we're
 * coming from, moved onto the navmesh. Positions are at eye height.
 */
export function landing(camera: THREE.PerspectiveCamera, poi: Poi, guard: NavmeshGuard | null) {
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
  const endRotation = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().lookAt(position, target, camera.up),
  );
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
