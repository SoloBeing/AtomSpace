import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';

const listen = { addEventListener: () => {} };
vi.stubGlobal('document', listen);
vi.stubGlobal('window', { ...listen, matchMedia: () => ({ matches: false }) });
const { FirstPersonControls } = await import('./firstPerson');

const FRAME = 1 / 60;

function walker() {
  const camera = new THREE.PerspectiveCamera(); // looks down -Z
  return new FirstPersonControls(camera, listen as unknown as HTMLElement);
}

const speedAfter = (controls: InstanceType<typeof FirstPersonControls>, seconds: number) => {
  let step = new THREE.Vector3();
  for (let t = 0; t < seconds; t += FRAME) step = controls.desiredMove(FRAME);
  return step.length() / FRAME;
};

describe('FirstPersonControls walking', () => {
  it('eases up to walking speed instead of jumping to it', () => {
    const controls = walker();
    controls.setStickMove(0, 1);
    expect(speedAfter(controls, FRAME)).toBeLessThan(0.3);
    expect(speedAfter(controls, 1)).toBeCloseTo(1.4, 2);
  });

  it('eases to a stop and then stays still', () => {
    const controls = walker();
    controls.setStickMove(0, 1);
    speedAfter(controls, 1);
    controls.setStickMove(0, 0);
    expect(speedAfter(controls, FRAME)).toBeGreaterThan(1);
    expect(speedAfter(controls, 1)).toBe(0);
  });

  it('steps back and sideways slower than forwards', () => {
    const back = walker();
    back.setStickMove(0, -1);
    expect(speedAfter(back, 1)).toBeCloseTo(1.4 * 0.6, 2);
    const side = walker();
    side.setStickMove(1, 0);
    expect(speedAfter(side, 1)).toBeCloseTo(1.4 * 0.8, 2);
  });

  it('bobs the head while walking and not when standing', () => {
    const controls = walker();
    expect(controls.headBob(0, FRAME).length()).toBe(0);
    const bobs = Array.from({ length: 30 }, () => controls.headBob(1.4 * FRAME, FRAME));
    const lowest = Math.min(...bobs.map((b) => b.y));
    expect(lowest).toBeLessThan(-0.015);
    expect(lowest).toBeGreaterThanOrEqual(-0.025);
  });
});
