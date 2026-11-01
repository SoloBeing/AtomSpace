import { describe, expect, it } from 'vitest';
import { nearestIndex } from './worldCarousel';

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
