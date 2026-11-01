import { describe, expect, it } from 'vitest';
import { nearestIndex, wheelStep } from './worldCarousel';

describe('wheelStep', () => {
  it('moves forward on scroll down and back on scroll up', () => {
    expect(wheelStep(0, 100)).toBe(1);
    expect(wheelStep(0, -100)).toBe(-1);
  });

  it('counts small trackpad movements too', () => {
    expect(wheelStep(0, 2)).toBe(1);
  });

  it('leaves mostly sideways scrolling to the track', () => {
    expect(wheelStep(40, 10)).toBe(0);
    expect(wheelStep(0, 0)).toBe(0);
  });
});

describe('nearestIndex', () => {
  it('finds the closest value', () => {
    expect(nearestIndex([100, 300, 500], 340)).toBe(1);
  });

  it('holds the ends when scrolled past them', () => {
    expect(nearestIndex([100, 300, 500], -50)).toBe(0);
    expect(nearestIndex([100, 300, 500], 900)).toBe(2);
  });

  it('keeps the earlier one on a tie', () => {
    expect(nearestIndex([100, 300], 200)).toBe(0);
  });
});
