import { DropInViewer, SceneRevealMode } from '@mkkellogg/gaussian-splats-3d';
import * as THREE from 'three';
import { isTouchDevice } from '../controls/touch';
import type { SceneBundle } from './loader';

export interface SplatViewer {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  onFrame(callback: (dtSeconds: number) => void): void;
  /** Resolves with the world point on the splat under the next canvas click, or null if it hit nothing. */
  pickNextClick(): Promise<THREE.Vector3 | null>;
  /** True while pickNextClick is waiting: other click handlers should leave that click alone. */
  readonly picking: boolean;
}

/**
 * Renders bundle.splatUrl with @mkkellogg/gaussian-splats-3d inside our own Three.js scene,
 * placing the splat with manifest.world_from_recon so it lines up with the navmesh and POIs.
 */
export async function createSplatViewer(
  container: HTMLElement,
  bundle: SceneBundle,
  startPosition: THREE.Vector3,
  onProgress?: (percent: number) => void,
): Promise<SplatViewer> {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  // Overlapping splats are expensive per pixel, and phones have very dense screens.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, isTouchDevice() ? 1.5 : 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(70, container.clientWidth / container.clientHeight, 0.05, 200);
  camera.position.copy(startPosition); // the first splat sort runs from here

  const callbacks: ((dtSeconds: number) => void)[] = [];
  const clock = new THREE.Clock();
  const startRenderLoop = () =>
    renderer.setAnimationLoop(() => {
      const dt = clock.getDelta();
      for (const callback of callbacks) callback(dt);
      renderer.render(scene, camera);
    });

  // Shared memory would need COOP/COEP headers on whatever host serves the viewer.
  const splats = new DropInViewer({
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: false,
    sceneRevealMode: SceneRevealMode.Instant,
  });
  scene.add(splats);
  // Before loading: drop-in mode initialises on its first rendered frame, and a scene that
  // finishes loading before then never gets its first sort, so it draws nothing.
  startRenderLoop();

  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  new THREE.Matrix4().fromArray(bundle.manifest.world_from_recon).decompose(position, rotation, scale);
  await splats.addSplatScene(bundle.splatUrl, {
    showLoadingUI: false,
    splatAlphaRemovalThreshold: 5,
    position: position.toArray(),
    rotation: rotation.toArray(),
    scale: scale.toArray(),
    onProgress: (percent) => onProgress?.(percent),
  });

  const pickAt = (event: MouseEvent): THREE.Vector3 | null => {
    const rect = renderer.domElement.getBoundingClientRect();
    const { raycaster, splatMesh } = splats.viewer;
    camera.updateMatrixWorld();
    raycaster.setFromCameraAndScreenPosition(
      camera,
      new THREE.Vector2(event.clientX - rect.left, event.clientY - rect.top),
      new THREE.Vector2(rect.width, rect.height),
    );
    const hits: { origin: THREE.Vector3 }[] = [];
    raycaster.intersectSplatMesh(splatMesh, hits);
    return hits.length > 0 ? hits[0].origin.clone() : null;
  };

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  let picking = false;
  return {
    scene,
    camera,
    domElement: renderer.domElement,
    onFrame: (callback) => callbacks.push(callback),
    pickNextClick: () =>
      new Promise((resolve) => {
        document.exitPointerLock();
        picking = true;
        // Capture phase, so this click doesn't also start mouse look.
        const onClick = (event: MouseEvent) => {
          event.stopImmediatePropagation();
          renderer.domElement.removeEventListener('click', onClick, true);
          picking = false;
          resolve(pickAt(event));
        };
        renderer.domElement.addEventListener('click', onClick, true);
      }),
    get picking() {
      return picking;
    },
  };
}
