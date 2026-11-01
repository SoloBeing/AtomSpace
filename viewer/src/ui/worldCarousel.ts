import { thumbnailUrl, type World } from '../scene/loader';

/** The index of the value closest to target. */
export function nearestIndex(values: number[], target: number): number {
  let best = 0;
  values.forEach((value, i) => {
    if (Math.abs(value - target) < Math.abs(values[best] - target)) best = i;
  });
  return best;
}

function createCard(world: World): HTMLAnchorElement {
  // A plain link, so opening a world is a fresh page: it can be shared, and Back returns here.
  const card = document.createElement('a');
  card.className = 'world-card';
  card.href = `?scene=${encodeURIComponent(world.scene)}`;
  const image = document.createElement('img');
  image.src = thumbnailUrl(world);
  image.alt = '';
  image.addEventListener('error', () => image.remove()); // the card's own background stands in
  const name = document.createElement('span');
  name.textContent = world.name;
  card.append(image, name);
  return card;
}

function createArrow(label: string, symbol: string, onClick: () => void): HTMLButtonElement {
  const arrow = document.createElement('button');
  arrow.type = 'button';
  arrow.setAttribute('aria-label', label);
  arrow.textContent = symbol;
  arrow.addEventListener('click', onClick);
  return arrow;
}

/** Full-screen chooser: swipe, scroll, or use the arrows, then open a world. */
export function mountWorldCarousel(container: HTMLElement, worlds: World[]): void {
  const root = document.createElement('main');
  root.className = 'worlds';
  const title = document.createElement('h1');
  title.textContent = 'Choose a world';
  const track = document.createElement('div');
  track.className = 'worlds-track';
  const cards = worlds.map(createCard);
  track.append(...cards);

  let active = 0;
  const setActive = (index: number) => {
    active = index;
    cards.forEach((card, i) => card.classList.toggle('active', i === index));
    previous.disabled = index === 0;
    next.disabled = index === cards.length - 1;
  };
  const go = (index: number) => {
    const target = Math.max(0, Math.min(index, cards.length - 1));
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    cards[target].scrollIntoView({ behavior: still ? 'auto' : 'smooth', inline: 'center', block: 'nearest' });
    cards[target].focus({ preventScroll: true });
    setActive(target);
  };

  const previous = createArrow('Previous world', '‹', () => go(active - 1));
  const next = createArrow('Next world', '›', () => go(active + 1));
  const carousel = document.createElement('div');
  carousel.className = 'worlds-carousel';
  carousel.append(previous, track, next);
  root.append(title, carousel);
  container.append(root);

  track.addEventListener('scroll', () => {
    const centers = cards.map((card) => card.offsetLeft + card.offsetWidth / 2);
    setActive(nearestIndex(centers, track.scrollLeft + track.clientWidth / 2));
  });
  root.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') go(active - 1);
    else if (event.key === 'ArrowRight') go(active + 1);
  });

  setActive(0);
  cards[0].focus({ preventScroll: true }); // arrow keys and Enter work straight away
}

/** The way back to the chooser from inside a world. */
export function mountWorldsLink(container: HTMLElement): void {
  const link = document.createElement('a');
  link.className = 'worlds-link';
  link.href = location.pathname;
  link.textContent = '‹ Worlds';
  container.append(link);
}
