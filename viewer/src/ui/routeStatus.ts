const ARRIVED_SHOWN_MS = 3000;

/** "4.2 m" up close, "12 m" further away: tenths stop meaning much past ten metres. */
export function formatDistance(metres: number): string {
  return metres < 10 ? `${metres.toFixed(1)} m` : `${Math.round(metres)} m`;
}

export interface RouteStatus {
  route(name: string, metres: number): void;
  arrived(name: string): void;
  hide(): void;
}

/**
 * A banner for the search route: where you're headed and how far is left, with a button to cancel,
 * then "You've arrived".
 */
export function mountRouteStatus(container: HTMLElement, onCancel: () => void): RouteStatus {
  const root = document.createElement('div');
  root.className = 'route-status';
  root.hidden = true;
  // Only the text is a live region, so the button isn't read out with every update.
  const text = document.createElement('span');
  text.setAttribute('role', 'status');
  const label = document.createElement('span');
  const distance = document.createElement('span');
  // It changes with every step; screen readers hear the destination and the arrival instead.
  distance.setAttribute('aria-hidden', 'true');
  text.append(label, distance);
  const cancel = document.createElement('button');
  cancel.type = 'button';
  cancel.textContent = '✕';
  cancel.title = 'Cancel route';
  cancel.setAttribute('aria-label', 'Cancel route');
  cancel.addEventListener('click', onCancel);
  root.append(text, cancel);
  container.append(root);

  let hideTimer = 0;
  const setText = (element: HTMLElement, text: string) => {
    if (element.textContent !== text) element.textContent = text;
  };

  return {
    route(name, metres) {
      clearTimeout(hideTimer);
      root.hidden = false;
      root.classList.remove('arrived');
      cancel.hidden = false;
      setText(label, `To ${name}`);
      setText(distance, ` · ${formatDistance(metres)}`);
    },
    arrived(name) {
      root.hidden = false;
      root.classList.add('arrived');
      cancel.hidden = true; // nothing left to cancel
      setText(label, `You've arrived at ${name}`);
      setText(distance, '');
      hideTimer = window.setTimeout(() => (root.hidden = true), ARRIVED_SHOWN_MS);
    },
    hide() {
      clearTimeout(hideTimer);
      root.hidden = true;
    },
  };
}
