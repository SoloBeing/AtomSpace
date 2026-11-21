import type { Poi } from '../scene/loader';

export interface PoiCard {
  /** Opens the card for `poi`; `detail` is a line under the name, such as the walking distance. */
  show(poi: Poi, detail: string): void;
  hide(): void;
}

/** The card a clicked pin opens: what it is, how far, and a button to get directions there. */
export function mountPoiCard(container: HTMLElement, onDirections: (poi: Poi) => void): PoiCard {
  const root = document.createElement('section');
  root.className = 'poi-card';
  root.setAttribute('role', 'dialog');
  root.hidden = true;
  const title = document.createElement('h2');
  title.id = 'poi-card-title';
  root.setAttribute('aria-labelledby', title.id);
  const category = document.createElement('p');
  category.className = 'category';
  const detail = document.createElement('p');
  const directions = document.createElement('button');
  directions.type = 'button';
  directions.textContent = 'Directions';
  const close = document.createElement('button');
  close.type = 'button';
  close.className = 'close';
  close.textContent = '✕';
  close.title = 'Close';
  close.setAttribute('aria-label', 'Close');
  root.append(close, title, category, detail, directions);
  container.append(root);

  let current: Poi | null = null;
  const hide = () => {
    current = null;
    root.hidden = true;
  };

  directions.addEventListener('click', () => {
    if (current) onDirections(current);
  });
  close.addEventListener('click', hide);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && current) hide();
  });

  return {
    show(poi, text) {
      current = poi;
      title.textContent = poi.name;
      category.textContent = poi.category ?? '';
      category.hidden = !poi.category;
      detail.textContent = text;
      detail.hidden = !text;
      root.hidden = false;
      directions.focus({ preventScroll: true }); // Enter goes straight there
    },
    hide,
  };
}
