import * as THREE from 'three';
import type { Vec3 } from '../scene/loader';

const WIDTH_M = 0.08;
const LIFT_M = 0.03; // above the navmesh overlay, so both can be shown
const COLOR = 0xff8a3d;

export interface RouteLine {
  show(points: Vec3[]): void;
  clear(): void;
}

/** A flat ribbon on the floor along a route, with round joints so corners don't show gaps. */
export function createRouteLine(scene: THREE.Scene): RouteLine {
  const material = new THREE.MeshBasicMaterial({
    color: COLOR,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const joint = new THREE.CircleGeometry(WIDTH_M / 2, 16).rotateX(-Math.PI / 2);
  const group = new THREE.Group();
  scene.add(group);

  const clear = () => {
    for (const child of group.children as THREE.Mesh[]) {
      if (child.geometry !== joint) child.geometry.dispose();
    }
    group.clear();
  };

  const show = (points: Vec3[]) => {
    clear();
    const lifted = points.map(([x, y, z]) => new THREE.Vector3(x, y + LIFT_M, z));
    const positions: number[] = [];
    for (let i = 1; i < lifted.length; i++) {
      const a = lifted[i - 1];
      const b = lifted[i];
      const side = new THREE.Vector3(-(b.z - a.z), 0, b.x - a.x).normalize().multiplyScalar(WIDTH_M / 2);
      if (!Number.isFinite(side.x)) continue; // zero-length step
      const [a1, a2, b1, b2] = [a.clone().add(side), a.clone().sub(side), b.clone().add(side), b.clone().sub(side)];
      positions.push(...a1.toArray(), ...a2.toArray(), ...b1.toArray(), ...a2.toArray(), ...b2.toArray(), ...b1.toArray());
    }
    const ribbon = new THREE.BufferGeometry();
    ribbon.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    group.add(new THREE.Mesh(ribbon, material));
    for (const point of lifted) {
      const dot = new THREE.Mesh(joint, material);
      dot.position.copy(point);
      group.add(dot);
    }
    group.traverse((object) => {
      object.renderOrder = 6; // over the navmesh overlay, under POI labels
    });
  };

  return { show, clear };
}
