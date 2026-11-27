export interface LoadingScreen {
  setProgress(percent: number): void;
  remove(): void;
}

/**
 * Full-screen title and progress bar, covering the page until `what` has loaded. `throughPortal`
 * gives arriving through a doorway its own look, so it reads as moving on, not starting over.
 */
export function mountLoadingScreen(container: HTMLElement, what = 'scene', throughPortal = false): LoadingScreen {
  const root = document.createElement('div');
  root.className = throughPortal ? 'loading portal' : 'loading';
  const title = document.createElement('h1');
  title.textContent = throughPortal ? 'Entering the next room' : 'AtomSpace';
  const bar = document.createElement('div');
  bar.className = 'loading-bar';
  bar.setAttribute('role', 'progressbar');
  bar.setAttribute('aria-label', `Loading ${what}`);
  bar.setAttribute('aria-valuemin', '0');
  bar.setAttribute('aria-valuemax', '100');
  const fill = document.createElement('div');
  bar.append(fill);
  const status = document.createElement('p');
  status.textContent = `Loading ${what}…`;
  root.append(title, bar, status);
  container.append(root);

  return {
    setProgress: (percent) => {
      const value = Math.round(Math.min(percent, 100));
      fill.style.width = `${value}%`;
      bar.setAttribute('aria-valuenow', String(value));
      // The file is down at 100%, but the splats still have to be processed before they appear.
      status.textContent = value < 100 ? `Loading ${what}… ${value}%` : `Preparing ${what}…`;
    },
    remove: () => root.remove(),
  };
}
