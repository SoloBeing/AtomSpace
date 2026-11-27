const TOUCH_CONTROLS: [keys: string[], action: string][] = [
  [['Left thumb'], 'Walk'],
  [['Right thumb'], 'Look'],
];

const MOUSE_CONTROLS: [keys: string[], action: string][] = [
  [['W', 'A', 'S', 'D'], 'Walk'],
  [['Shift'], 'Sprint'],
  [['Space'], 'Jump'],
  [['M'], 'Mute sounds'],
  [['Mouse'], 'Look'],
  [['Esc'], 'Free cursor'],
];

/**
 * Shows how to move around. With a mouse it fades once the mouse is captured and returns when
 * it's released; on touch it fades for good at the first touch.
 */
export function mountControlsHint(container: HTMLElement, lockTarget: HTMLElement, touch: boolean): void {
  const root = document.createElement('div');
  root.className = touch ? 'controls-hint touch' : 'controls-hint';
  const prompt = document.createElement('p');
  prompt.textContent = touch ? 'Drag on the screen to move' : 'Click the scene to start';
  const list = document.createElement('dl');
  for (const [keys, action] of touch ? TOUCH_CONTROLS : MOUSE_CONTROLS) {
    const term = document.createElement('dt');
    term.append(
      ...keys.map((key) => {
        const cap = document.createElement('kbd');
        cap.textContent = key;
        return cap;
      }),
    );
    const description = document.createElement('dd');
    description.textContent = action;
    list.append(term, description);
  }
  root.append(prompt, list);
  container.append(root);

  if (touch) {
    lockTarget.addEventListener('pointerdown', () => root.classList.add('locked'), { once: true });
  } else {
    document.addEventListener('pointerlockchange', () => {
      root.classList.toggle('locked', document.pointerLockElement === lockTarget);
    });
  }
}
