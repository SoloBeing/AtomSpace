import { describe, expect, it } from 'vitest';
import { parseWorlds } from './loader';

describe('parseWorlds', () => {
  it('reads the list', () => {
    const worlds = [{ scene: 'friend-terrace', name: "Friend's Terrace" }];
    expect(parseWorlds({ format_version: 1, worlds })).toEqual(worlds);
  });

  it('refuses an empty list, so the page can say what to do', () => {
    expect(() => parseWorlds({ format_version: 1, worlds: [] })).toThrow();
  });

  it('refuses anything else, such as an HTML fallback page', () => {
    expect(() => parseWorlds('<!doctype html>')).toThrow();
  });

  it('refuses a scene name that the scene server would not accept', () => {
    expect(() => parseWorlds({ worlds: [{ scene: '../secret', name: 'x' }] })).toThrow();
  });
});
