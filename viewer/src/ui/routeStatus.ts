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

/** A banner for the search route: where you're headed and how far is left, then "You've arrived". */
export function mountRouteStatus(container: HTMLElement): RouteStatus {
  const root = document.createElement('div');
  root.className = 'route-status';
  root.setAttribute('role', 'status');
  root.hidden = true;
  const label = document.createElement('span');
  const distance = document.createElement('span');
  // It changes with every step; screen readers hear the destination and the arrival instead.
  distance.setAttribute('aria-hidden', 'true');
  root.append(label, distance);
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
      setText(label, `To ${name}`);
      setText(distance, ` · ${formatDistance(metres)}`);
    },
    arrived(name) {
      root.hidden = false;
      root.classList.add('arrived');
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
