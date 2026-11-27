import type { Poi } from '../scene/loader';

/** How close (horizontally, metres) you must stand to a doorway to be offered to go through. */
export const PORTAL_REACH_M = 1.5;
const FADE_MS = 600;

/** The page URL for walking through `poi`'s portal, keeping admin mode if it's on. */
export function portalUrl(poi: Poi, current: URLSearchParams): string {
  const next = new URLSearchParams({ scene: poi.portal!.scene, via: '1' });
  if (poi.portal!.arrive) next.set('arrive', poi.portal!.arrive);
  if (current.has('admin')) next.set('admin', '');
  return `?${next}`;
}

/** The nearest portal within reach of a floor point (x, z), or null. */
export function portalInReach(pois: Poi[], x: number, z: number): Poi | null {
  let best: Poi | null = null;
  let bestDistance = PORTAL_REACH_M;
  for (const poi of pois) {
    if (!poi.portal) continue;
    const distance = Math.hypot(poi.position[0] - x, poi.position[2] - z);
    if (distance <= bestDistance) {
      bestDistance = distance;
      best = poi;
    }
  }
  return best;
}

export interface PortalPrompt {
  /** Shows the prompt for `poi`, or hides it when null. Cheap to call every frame. */
  offer(poi: Poi | null): void;
}

/**
 * "Press E to enter …" while standing at a doorway. It is a button too, so a tap works on phones.
 * Going through fades the screen out before `onEnter` swaps the scene.
 */
export function mountPortalPrompt(container: HTMLElement, touch: boolean, onEnter: (poi: Poi) => void): PortalPrompt {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'portal-prompt';
  button.hidden = true;
  const fade = document.createElement('div');
  fade.className = 'portal-fade';
  container.append(button, fade);

  let current: Poi | null = null;
  let entering = false;
  const enter = () => {
    if (!current || entering) return;
    entering = true;
    const poi = current;
    fade.classList.add('on');
    setTimeout(() => onEnter(poi), FADE_MS);
  };
  button.addEventListener('click', enter);
  document.addEventListener('keydown', (event) => {
    if (event.code === 'KeyE' && !(event.target instanceof HTMLInputElement)) enter();
  });

  return {
    offer(poi) {
      if (poi === current) return;
      current = poi;
      button.hidden = !poi;
      if (poi) button.textContent = `${touch ? 'Tap' : 'Press E'} to enter: ${poi.name}`;
    },
  };
}
