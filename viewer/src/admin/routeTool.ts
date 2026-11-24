import { type NavmeshGuard, toFloor } from '../controls/navmeshGuard';
import { type Pathfinder, routeLength } from '../nav/pathfinder';
import { createRouteLine } from '../nav/routeLine';
import type { SplatViewer } from '../scene/splatRenderer';
import { adminNote, adminPanel } from './panel';

/** Admin: click anywhere in the scene and see the walking route there from where you stand. */
export function mountRouteTool(viewer: SplatViewer, nav: { guard: NavmeshGuard; pathfinder: Pathfinder } | null): void {
  const panel = adminPanel('Route');
  if (!nav) return adminNote(panel, 'Routes need a navmesh.');

  const line = createRouteLine(viewer.scene);
  const pick = document.createElement('button');
  pick.textContent = 'Route to a clicked point';
  const clear = document.createElement('button');
  clear.textContent = 'Clear route';
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  panel.append(pick, clear, status);

  pick.addEventListener('click', async () => {
    pick.textContent = 'Click where to go…';
    const hit = await viewer.pickNextClick();
    pick.textContent = 'Route to a clicked point';
    if (!hit) {
      status.textContent = 'That click missed the scene, try again.';
      return;
    }
    // A click on a wall or a table means "take me there": the nearest floor you can stand on.
    const goal = nav.guard.nearestPosition(hit.x, hit.z);
    const path = nav.pathfinder.findPath(toFloor(viewer.camera.position), toFloor(goal));
    if (!path) {
      line.clear();
      status.textContent = 'No walking route from here to there.';
      return;
    }
    line.show(path);
    const turns = path.length - 2;
    status.textContent = `${routeLength(path).toFixed(1)} m, ${turns} ${turns === 1 ? 'turn' : 'turns'}.`;
  });

  clear.addEventListener('click', () => {
    line.clear();
    status.textContent = '';
  });
}
