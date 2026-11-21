import * as THREE from 'three';
import type { Poi } from '../scene/loader';

const LABEL_HEIGHT = 0.05; // fraction of the view: readable from across a room and not huge up close
const FONT_PX = 40;
const PADDING_PX = 16;
const LIFT_M = 0.2; // the label floats above the point it marks

export interface Pins {
  update(pois: Poi[]): void;
  /** The POI whose label is under this point of the view (normalised device coordinates), nearest first. */
  at(ndc: THREE.Vector2, camera: THREE.Camera): Poi | null;
  dispose(): void;
}

function labelSprite(text: string): THREE.Sprite {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d')!;
  const font = `600 ${FONT_PX}px system-ui, sans-serif`;
  context.font = font;
  canvas.width = Math.ceil(context.measureText(text).width) + 2 * PADDING_PX;
  canvas.height = FONT_PX + 2 * PADDING_PX;
  context.font = font; // resizing a canvas resets its state
  context.fillStyle = 'rgba(17, 17, 17, 0.8)';
  context.beginPath();
  context.roundRect(0, 0, canvas.width, canvas.height, PADDING_PX);
  context.fill();
  context.fillStyle = '#fff';
  context.textBaseline = 'middle';
  context.fillText(text, PADDING_PX, canvas.height / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set((LABEL_HEIGHT * canvas.width) / canvas.height, LABEL_HEIGHT, 1);
  sprite.renderOrder = 10; // after the splats, which are drawn as transparent
  return sprite;
}

/** Billboard pins with name labels at each POI position. */
export function createPins(scene: THREE.Scene, pois: Poi[]): Pins {
  const group = new THREE.Group();
  scene.add(group);

  const clear = () => {
    for (const sprite of group.children as THREE.Sprite[]) {
      sprite.material.map?.dispose();
      sprite.material.dispose();
    }
    group.clear();
  };

  const update = (next: Poi[]) => {
    clear();
    for (const poi of next) {
      const sprite = labelSprite(poi.name);
      sprite.position.fromArray(poi.position);
      sprite.position.y += LIFT_M;
      sprite.userData.poi = poi;
      group.add(sprite);
    }
  };

  const raycaster = new THREE.Raycaster();
  update(pois);
  return {
    update,
    at: (ndc, camera) => {
      raycaster.setFromCamera(ndc, camera); // also gives sprites the camera they need to be hit
      const hit = raycaster.intersectObjects(group.children, false)[0];
      return hit ? (hit.object.userData.poi as Poi) : null;
    },
    dispose: () => {
      clear();
      scene.remove(group);
    },
  };
}
