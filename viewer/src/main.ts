import * as THREE from 'three';
import './ui/styles.css';
import { mountNavmeshOverlay } from './admin/navmeshOverlay';
import { mountPoiEditor } from './admin/poiEditor';
import { mountRouteTool } from './admin/routeTool';
import { mountScaleTool } from './admin/scaleTool';
import { FirstPersonControls } from './controls/firstPerson';
import { EYE_HEIGHT_M, NavmeshGuard } from './controls/navmeshGuard';
import { isTouchDevice, mountTouchControls } from './controls/touch';
import { Directions } from './nav/directions';
import { Pathfinder } from './nav/pathfinder';
import { createRouteLine } from './nav/routeLine';
import { flyTo } from './poi/flyTo';
import { createPins } from './poi/pins';
import { createPoiSearch } from './poi/search';
import { loadScene, loadWorlds, preloadThumbnails, type Vec3 } from './scene/loader';
import { createSplatViewer } from './scene/splatRenderer';
import { mountControlsHint } from './ui/controlsHint';
import { mountFpsCounter } from './ui/fpsCounter';
import { mountLoadingScreen } from './ui/loadingScreen';
import { mountSearchBar } from './ui/searchBar';
import { mountWorldCarousel, mountWorldsLink } from './ui/worldCarousel';

async function start(app: HTMLElement, sceneName: string, admin: boolean) {
  const loading = mountLoadingScreen(app);
  const bundle = await loadScene(sceneName);
  const guard = bundle.navmesh ? new NavmeshGuard(bundle.navmesh) : null;
  // Without a navmesh the first camera is at the origin, already at its own height.
  const startPosition = guard ? guard.spawnPosition() : new THREE.Vector3();
  const viewer = await createSplatViewer(app, bundle, startPosition, loading.setProgress);
  loading.remove();

  const controls = new FirstPersonControls(viewer.camera, viewer.domElement);
  const touch = isTouchDevice();
  mountControlsHint(app, viewer.domElement, touch);
  if (touch) mountTouchControls(app, viewer.domElement, controls);
  viewer.onFrame(mountFpsCounter(app)); // mounted before search, so results draw over it
  const pins = createPins(viewer.scene, bundle.pois);
  let search = createPoiSearch(bundle.pois);
  const pathfinder = bundle.navmesh ? new Pathfinder(bundle.navmesh) : null;
  const directions = pathfinder ? new Directions(pathfinder, createRouteLine(viewer.scene)) : null;
  const feet = (p: THREE.Vector3): Vec3 => [p.x, p.y - EYE_HEIGHT_M, p.z];
  let flying = false;
  let trip = 0;
  mountSearchBar(
    app,
    (query, limit) => search(query, limit),
    (poi) => {
      // With a navmesh, show the way right up to the thing (as near as the navmesh allows) and let the user walk it.
      if (directions && guard) {
        const end = guard.nearestPosition(poi.position[0], poi.position[2]);
        if (directions.start(feet(viewer.camera.position), feet(end))) return;
      }
      // No navmesh, or no walking route to this POI: nothing to draw, so fly there.
      flying = true;
      const thisTrip = ++trip;
      void flyTo(viewer.camera, poi, guard).then(() => {
        // A newer trip may have cancelled this one; only the latest hands control back.
        if (thisTrip === trip) flying = false;
      });
    },
  );

  if (admin) {
    mountNavmeshOverlay(viewer, bundle.navmesh);
    mountRouteTool(viewer, guard && pathfinder ? { guard, pathfinder } : null);
    mountScaleTool(viewer, bundle);
    mountPoiEditor(viewer, bundle, (pois) => {
      pins.update(pois);
      search = createPoiSearch(pois);
    });
  } else {
    mountWorldsLink(app); // shares the top right corner with the admin panels
  }

  viewer.onFrame((dt) => {
    if (flying) return;
    const position = viewer.camera.position;
    const delta = controls.desiredMove(dt);
    if (delta.lengthSq() === 0) return;
    position.copy(guard ? guard.tryMove(position, delta) : position.clone().add(delta));
    directions?.update(feet(position));
  });
}

async function chooseWorld(app: HTMLElement) {
  const loading = mountLoadingScreen(app, 'worlds');
  const worlds = await loadWorlds();
  await preloadThumbnails(worlds, loading.setProgress);
  loading.remove();
  mountWorldCarousel(app, worlds);
}

const app = document.getElementById('app')!;
const params = new URLSearchParams(location.search);
const scene = params.get('scene');

if (!scene) {
  chooseWorld(app).catch((err) => {
    console.error(err);
    app.textContent = 'No worlds found. Add ?scene=<name> to the URL.';
  });
} else {
  start(app, scene, params.has('admin')).catch((err) => {
    console.error(err);
    app.textContent = `Could not load scene "${scene}": ${err.message}`;
  });
}
