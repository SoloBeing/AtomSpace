import { thumbnailUrl, type World } from '../scene/loader';

/** The index of the value closest to target. */
export function nearestIndex(values: number[], target: number): number {
  let best = 0;
  values.forEach((value, i) => {
    if (Math.abs(value - target) < Math.abs(values[best] - target)) best = i;
  });
  return best;
}

/** Which way a wheel event moves the carousel. Sideways scrolling is left to the track itself. */
export function wheelStep(deltaX: number, deltaY: number): -1 | 0 | 1 {
  if (deltaY === 0 || Math.abs(deltaX) >= Math.abs(deltaY)) return 0;
  return deltaY > 0 ? 1 : -1;
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

// One wheel flick sends many events, so one step per this long. Trackpads coast for a while.
const WHEEL_STEP_MS = 350;
// While go() animates the track, its scroll events report the cards it's passing, not the target.
const GO_SETTLE_MS = 500;

/** Full-screen chooser: swipe, scroll, or use the arrows or dots, then open a world. */
export function mountWorldCarousel(container: HTMLElement, worlds: World[]): void {
  const root = document.createElement('main');
  root.className = 'worlds';
  const title = document.createElement('h1');
  title.textContent = 'Choose a World';
  const track = document.createElement('div');
  track.className = 'worlds-track';
  const cards = worlds.map(createCard);
  track.append(...cards);

  let active = 0;
  let settlingUntil = 0;
  const setActive = (index: number) => {
    active = index;
    cards.forEach((card, i) => card.classList.toggle('active', i === index));
    dots.forEach((dot, i) => {
      if (i === index) dot.setAttribute('aria-current', 'true');
      else dot.removeAttribute('aria-current');
    });
    previous.disabled = index === 0;
    next.disabled = index === cards.length - 1;
  };
  const go = (index: number) => {
    const target = Math.max(0, Math.min(index, cards.length - 1));
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    cards[target].scrollIntoView({ behavior: still ? 'auto' : 'smooth', inline: 'center', block: 'nearest' });
    cards[target].focus({ preventScroll: true });
    settlingUntil = performance.now() + GO_SETTLE_MS;
    setActive(target);
  };

  const previous = createArrow('Previous world', '‹', () => go(active - 1));
  const next = createArrow('Next world', '›', () => go(active + 1));
  const carousel = document.createElement('div');
  carousel.className = 'worlds-carousel';
  carousel.append(previous, track, next);

  const dots = worlds.map((world, i) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.setAttribute('aria-label', `Show ${world.name}`);
    dot.addEventListener('click', () => go(i));
    return dot;
  });
  const dotRow = document.createElement('div');
  dotRow.className = 'worlds-dots';
  dotRow.append(...dots);

  root.append(title, carousel, dotRow);
  container.append(root);

  let nextWheelStep = 0;
  root.addEventListener('wheel', (event) => {
    const step = wheelStep(event.deltaX, event.deltaY);
    if (step === 0 || event.timeStamp < nextWheelStep) return;
    nextWheelStep = event.timeStamp + WHEEL_STEP_MS;
    go(active + step);
  });

  track.addEventListener('scroll', () => {
    if (performance.now() < settlingUntil) return;
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
