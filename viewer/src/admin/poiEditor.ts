import type * as THREE from 'three';
import type { Poi, SceneBundle, Vec3 } from '../scene/loader';
import type { SplatViewer } from '../scene/splatRenderer';
import { adminNote, adminPanel } from './panel';

const vec3 = (v: THREE.Vector3): Vec3 => [v.x, v.y, v.z];

function uniqueId(name: string, taken: Poi[]): string {
  const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'poi';
  let id = base;
  for (let n = 2; taken.some((poi) => poi.id === id); n++) id = `${base}-${n}`;
  return id;
}

async function savePois(scene: string, pois: Poi[]): Promise<void> {
  const res = await fetch(`/api/scenes/${encodeURIComponent(scene)}/pois`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format_version: 1, pois }),
  });
  if (!res.ok) throw new Error(await res.text());
}

/**
 * Admin: click the scene to place a POI, name it, set the landing view, then save pois.json.
 * The landing view is wherever the camera is when the POI is saved. Saving needs `npm run dev`.
 */
export function mountPoiEditor(viewer: SplatViewer, bundle: SceneBundle, onChange: (pois: Poi[]) => void): void {
  const panel = adminPanel('Points of interest');
  if (bundle.manifest.scale.source === 'unscaled') {
    // POIs live in world space, which moves once the scale and floor are set.
    return adminNote(panel, 'Set the scale and build the navmesh first.');
  }

  let pois = bundle.pois;
  let placed: THREE.Vector3 | null = null;

  const place = document.createElement('button');
  const name = document.createElement('input');
  name.placeholder = 'Name';
  const aliases = document.createElement('input');
  aliases.placeholder = 'Other names, comma separated';
  const category = document.createElement('input');
  category.placeholder = 'Category';
  const save = document.createElement('button');
  save.textContent = 'Save POI from this view';
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  const list = document.createElement('ul');
  panel.append(place, name, aliases, category, save, status, list);

  const persist = async (next: Poi[], done: string): Promise<boolean> => {
    try {
      await savePois(bundle.manifest.scene, next);
    } catch (err) {
      status.textContent = `Not saved: ${(err as Error).message}`;
      return false;
    }
    pois = next;
    onChange(pois);
    status.textContent = done;
    renderList();
    return true;
  };

  const renderList = () => {
    list.replaceChildren(
      ...pois.map((poi) => {
        const item = document.createElement('li');
        item.textContent = poi.name;
        const remove = document.createElement('button');
        remove.textContent = 'Delete';
        remove.setAttribute('aria-label', `Delete ${poi.name}`);
        remove.addEventListener('click', () => {
          void persist(pois.filter((p) => p !== poi), `Deleted ${poi.name}.`);
        });
        item.append(remove);
        return item;
      }),
    );
  };

  const refreshPlace = () => {
    place.textContent = placed ? 'Point placed, click to move it' : 'Place a point';
  };

  place.addEventListener('click', async () => {
    place.textContent = 'Click the scene…';
    const hit = await viewer.pickNextClick();
    if (hit) placed = hit;
    status.textContent = hit ? '' : 'That click missed the scene, try again.';
    refreshPlace();
  });

  save.addEventListener('click', async () => {
    const poiName = name.value.trim();
    if (!placed || !poiName) {
      status.textContent = 'Place a point and give it a name first.';
      return;
    }
    const poi: Poi = {
      id: uniqueId(poiName, pois),
      name: poiName,
      position: vec3(placed),
      view: { position: vec3(viewer.camera.position), target: vec3(placed) },
      source: 'manual',
    };
    const otherNames = aliases.value.split(',').map((s) => s.trim()).filter(Boolean);
    if (otherNames.length > 0) poi.aliases = otherNames;
    if (category.value.trim()) poi.category = category.value.trim();

    if (await persist([...pois, poi], `Saved ${poiName}.`)) {
      placed = null;
      name.value = aliases.value = category.value = '';
      refreshPlace();
    }
  });

  refreshPlace();
  renderList();
}
