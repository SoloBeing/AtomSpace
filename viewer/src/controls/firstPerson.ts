import * as THREE from 'three';
import { isTouchDevice } from './touch';

const WALK_SPEED_MPS = 1.4;

const SPRINT_MULTIPLIER = 2.5;
const BACKWARD_FACTOR = 0.6; // people step back and sideways slower than they walk forwards
const STRAFE_FACTOR = 0.8;
const EASE_SECONDS = 0.12; // speed reaches ~90% of the target in about 0.3 s, starting or stopping
const STEP_LENGTH_M = 0.7;
const BOB_HEIGHT_M = 0.025; // up-down, twice per stride (every footstep)
const BOB_SWAY_M = 0.012; // side to side, once per stride
const MOUSE_SENSITIVITY = 0.002;
const MAX_PITCH = Math.PI / 2 - 0.01;
const MAX_STEP_SECONDS = 0.1; // after a stall, don't jump a long way in one frame

/** WASD + pointer-lock mouse look, plus a touch stick fed in from outside. Rotates the camera itself; movement is only proposed, never applied. */
export class FirstPersonControls {
  private readonly keys = new Set<string>();
  private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');
  private readonly stick = new THREE.Vector2();
  private readonly velocity = new THREE.Vector3();
  private stepPhase = 0;
  private readonly reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

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

  /** Horizontal displacement the user is asking for this frame, in world meters. Eases in and out like a body, not a drone. */
  desiredMove(dtSeconds: number): THREE.Vector3 {
    const dt = Math.min(dtSeconds, MAX_STEP_SECONDS);
    const input = new THREE.Vector2(
      Number(this.keys.has('KeyD')) - Number(this.keys.has('KeyA')) + this.stick.x,
      Number(this.keys.has('KeyW')) - Number(this.keys.has('KeyS')) + this.stick.y,
    ).clampLength(0, 1);

    const facing = this.camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    const right = new THREE.Vector3(-facing.z, 0, facing.x);
    const sprinting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const speed = sprinting ? WALK_SPEED_MPS * SPRINT_MULTIPLIER : WALK_SPEED_MPS;
    const target = facing
      .multiplyScalar(input.y * (input.y < 0 ? BACKWARD_FACTOR : 1))
      .addScaledVector(right, input.x * STRAFE_FACTOR)
      .multiplyScalar(speed);

    this.velocity.lerp(target, 1 - Math.exp(-dt / EASE_SECONDS));
    if (target.lengthSq() === 0 && this.velocity.lengthSq() < 1e-4) this.velocity.set(0, 0, 0);
    return this.velocity.clone().multiplyScalar(dt);
  }

  /** Forget any momentum, e.g. after a fly-to moved the camera for us. */
  stop(): void {
    this.velocity.set(0, 0, 0);
  }

  /**
   * Camera offset for the footsteps behind `walkedMeters` of travel this frame: a dip each step and a
   * sway each stride, growing with speed so it fades out as walking eases to a stop.
   */
  headBob(walkedMeters: number, dtSeconds: number): THREE.Vector3 {
    if (this.reducedMotion || dtSeconds <= 0) return new THREE.Vector3();
    this.stepPhase = (this.stepPhase + (walkedMeters / STEP_LENGTH_M) * Math.PI) % (2 * Math.PI);
    const strength = Math.min(walkedMeters / dtSeconds / WALK_SPEED_MPS, 1);
    const facing = this.camera.getWorldDirection(new THREE.Vector3()).setY(0).normalize();
    const right = new THREE.Vector3(-facing.z, 0, facing.x);
    return right
      .multiplyScalar(BOB_SWAY_M * Math.sin(this.stepPhase))
      .setY(-BOB_HEIGHT_M * Math.abs(Math.sin(this.stepPhase)))
      .multiplyScalar(strength);
  }
}
