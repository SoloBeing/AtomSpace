import * as THREE from 'three';
import { isTouchDevice } from './touch';

const WALK_SPEED_MPS = 1.4;

const SPRINT_MULTIPLIER = 2.5;
const MOUSE_SENSITIVITY = 0.002;
const MAX_PITCH = Math.PI / 2 - 0.01;
const MAX_STEP_SECONDS = 0.1; // after a stall, don't jump a long way in one frame

/** WASD + pointer-lock mouse look, plus a touch stick fed in from outside. Rotates the camera itself; movement is only proposed, never applied. */
export class FirstPersonControls {
  private readonly keys = new Set<string>();
  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');
  private readonly stick = new THREE.Vector2();

  private readonly onClick = () => {
    // Phones have no pointer lock, and on iPhones the method doesn't exist at all.
    if (!isTouchDevice()) void this.domElement.requestPointerLock();
  };

  private readonly onMouseMove = (event: MouseEvent) => {
    if (document.pointerLockElement !== this.domElement) return;
    this.look(event.movementX * MOUSE_SENSITIVITY, event.movementY * MOUSE_SENSITIVITY);
  };

  private readonly onKeyDown = (event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement) return; // typing in the search bar isn't walking
    this.keys.add(event.code);
  };

  private readonly onKeyUp = (event: KeyboardEvent) => {
    this.keys.delete(event.code);
  };

  private readonly onBlur = () => {
    this.keys.clear();
  };

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    private readonly domElement: HTMLElement,
  ) {
    domElement.addEventListener('click', this.onClick);
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('keydown', this.onKeyDown);
    document.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
  }

  /** Turns the camera by these angles in radians: positive is right and down. */
  look(yaw: number, pitch: number): void {
    // Read back each time: flyTo also turns the camera.
    this.euler.setFromQuaternion(this.camera.quaternion);
    this.euler.y -= yaw;
    this.euler.x = THREE.MathUtils.clamp(this.euler.x - pitch, -MAX_PITCH, MAX_PITCH);
    this.camera.quaternion.setFromEuler(this.euler);
  }

  /** Touch stick position, each axis -1 to 1. Partway walks slower, unlike a key. */
  setStickMove(strafe: number, forward: number): void {
    this.stick.set(strafe, forward);
  }

  /** Horizontal displacement the user is asking for this frame, in world meters. */
  desiredMove(dtSeconds: number): THREE.Vector3 {
    const strafe = Number(this.keys.has('KeyD')) - Number(this.keys.has('KeyA')) + this.stick.x;
    const forward = Number(this.keys.has('KeyW')) - Number(this.keys.has('KeyS')) + this.stick.y;
    if (strafe === 0 && forward === 0) return new THREE.Vector3();

    const facing = this.camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    const right = new THREE.Vector3(-facing.z, 0, facing.x);
    const sprinting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const speed = sprinting ? WALK_SPEED_MPS * SPRINT_MULTIPLIER : WALK_SPEED_MPS;
    return facing
      .multiplyScalar(forward)
      .addScaledVector(right, strafe)
      .clampLength(0, 1)
      .multiplyScalar(speed * Math.min(dtSeconds, MAX_STEP_SECONDS));
  }
}
