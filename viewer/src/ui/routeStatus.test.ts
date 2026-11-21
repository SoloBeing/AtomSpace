import { describe, expect, it } from 'vitest';
import { formatDistance } from './routeStatus';

describe('formatDistance', () => {
  it('shows tenths up close and whole metres further away', () => {
    expect(formatDistance(4.24)).toBe('4.2 m');
    expect(formatDistance(0.04)).toBe('0.0 m');
    expect(formatDistance(9.94)).toBe('9.9 m');
    expect(formatDistance(12.6)).toBe('13 m');
  });
});
