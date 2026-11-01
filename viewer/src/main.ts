import * as THREE from 'three';
import './ui/styles.css';
import { mountPoiEditor } from './admin/poiEditor';
import { mountScaleTool } from './admin/scaleTool';
import { FirstPersonControls } from './controls/firstPerson';
import { NavmeshGuard } from './controls/navmeshGuard';
import { isTouchDevice, mountTouchControls } from './controls/touch';
import { flyTo } from './poi/flyTo';
import { createPins } from './poi/pins';
import { createPoiSearch } from './poi/search';
import { loadScene, loadWorlds, preloadThumbnails } from './scene/loader';
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
  let flying = false;
  mountSearchBar(
    app,
    (query, limit) => search(query, limit),
    (poi) => {
      flying = true;
      void flyTo(viewer.camera, poi, guard).then(() => {
        flying = false;
      });
    },
  );

  if (admin) {
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
    position.copy(guard ? guard.tryMove(position, delta) : position.clone().add(delta));
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
