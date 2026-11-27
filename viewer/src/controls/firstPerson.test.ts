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

  it('jumps in an arc about 0.4 m high and lands back on the floor', () => {
    const controls = walker();
    controls.jump();
    const heights = Array.from({ length: 90 }, () => controls.airborne(FRAME));
    expect(Math.max(...heights)).toBeCloseTo(0.4, 1);
    expect(heights.at(-1)).toBe(0);
    controls.jump(); // can jump again once landed
    expect(controls.airborne(FRAME)).toBeGreaterThan(0);
  });

  it('cannot jump again mid-air', () => {
    const controls = walker();
    controls.jump();
    for (let i = 0; i < 20; i++) controls.airborne(FRAME); // past the top (~0.29 s)
    const falling = controls.airborne(FRAME);
    controls.jump();
    expect(controls.airborne(FRAME)).toBeLessThan(falling);
  });

  it('sounds a footstep every 0.7 m and a jump and a landing', () => {
    const controls = walker();
    const heard: string[] = [];
    controls.onStep = () => heard.push('step');
    controls.onJump = () => heard.push('jump');
    controls.onLand = () => heard.push('land');
    for (let i = 0; i < 300; i++) controls.headBob(1.4 * FRAME, FRAME); // 7 m in 5 s
    expect(heard.filter((h) => h === 'step')).toHaveLength(10);
    heard.length = 0;
    controls.jump();
    for (let i = 0; i < 60; i++) {
      controls.airborne(FRAME);
      controls.headBob(1.4 * FRAME, FRAME); // no footsteps in the air
    }
    expect(heard.slice(0, heard.indexOf('land') + 1)).toEqual(['jump', 'land']); // steps resume only after landing
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
