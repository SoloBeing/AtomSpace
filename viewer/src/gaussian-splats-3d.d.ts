/** The parts of @mkkellogg/gaussian-splats-3d we use; the package ships no types. */
declare module '@mkkellogg/gaussian-splats-3d' {
  import type * as THREE from 'three';

  export const SceneRevealMode: { Default: number; Gradual: number; Instant: number };

  interface SplatHit {
    origin: THREE.Vector3;
  }

  interface InternalViewer {
    splatMesh: unknown;
    raycaster: {
      setFromCameraAndScreenPosition(camera: THREE.Camera, screen: THREE.Vector2, size: THREE.Vector2): void;
      intersectSplatMesh(mesh: unknown, hits: SplatHit[]): void;
    };
  }

  export class DropInViewer extends THREE.Group {
    constructor(options?: Record<string, unknown>);
    viewer: InternalViewer;
    addSplatScene(
      path: string,
      options?: {
        showLoadingUI?: boolean;
        splatAlphaRemovalThreshold?: number;
        position?: number[];
        rotation?: number[];
        scale?: number[];
        onProgress?: (percent: number, label: string, stage: number) => void;
      },
    ): Promise<void>;
  }
}
