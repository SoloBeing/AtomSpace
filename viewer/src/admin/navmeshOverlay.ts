import * as THREE from 'three';
import type { Navmesh } from '../scene/loader';
import type { SplatViewer } from '../scene/splatRenderer';
import { adminNote, adminPanel } from './panel';

const LIFT_M = 0.02; // just above the floor so it doesn't flicker against floor splats
const FILL = 0x3ccf6e;
const INNER_EDGE = 0x1f7a3f;
const OUTLINE = 0xffe14d;

export interface NavmeshLines {
  /** Triangle positions, 9 numbers per triangle. */
  fill: Float32Array;
  /** Edges shared by two polygons, 6 numbers per edge. */
  inner: Float32Array;
  /** Edges with no neighbour: where walking stops. */
  outline: Float32Array;
  areaM2: number;
}

/** Splits the navmesh into fill triangles, shared edges and outline edges, each edge once. */
export function navmeshLines(navmesh: Navmesh): NavmeshLines {
  const fill: number[] = [];
  const inner: number[] = [];
  const outline: number[] = [];
  let areaM2 = 0;
  const at = (i: number): number[] => {
    const [x, y, z] = navmesh.vertices[i];
    return [x, y + LIFT_M, z];
  };

  navmesh.polygons.forEach(({ v, n }, index) => {
    for (let i = 1; i < v.length - 1; i++) {
      const [a, b, c] = [navmesh.vertices[v[0]], navmesh.vertices[v[i]], navmesh.vertices[v[i + 1]]];
      fill.push(...at(v[0]), ...at(v[i]), ...at(v[i + 1]));
      areaM2 += Math.abs((b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0])) / 2;
    }
    v.forEach((start, i) => {
      const neighbour = n?.[i] ?? -1;
      const segment = [...at(start), ...at(v[(i + 1) % v.length])];
      if (neighbour === -1) outline.push(...segment);
      else if (neighbour > index) inner.push(...segment); // the lower-numbered side skips it
    });
  });
  return { fill: new Float32Array(fill), inner: new Float32Array(inner), outline: new Float32Array(outline), areaM2 };
}

function overlayGroup(lines: NavmeshLines): THREE.Group {
  const geometry = (positions: Float32Array) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return g;
  };
  // Splats don't write depth, so depth testing would hide nothing anyway; draw on top, after them.
  const common = { transparent: true, depthTest: false, depthWrite: false };
  const fill = new THREE.Mesh(
    geometry(lines.fill),
    new THREE.MeshBasicMaterial({ ...common, color: FILL, opacity: 0.28, side: THREE.DoubleSide }),
  );
  const inner = new THREE.LineSegments(
    geometry(lines.inner),
    new THREE.LineBasicMaterial({ ...common, color: INNER_EDGE, opacity: 0.6 }),
  );
  const outline = new THREE.LineSegments(geometry(lines.outline), new THREE.LineBasicMaterial({ ...common, color: OUTLINE }));
  const group = new THREE.Group();
  group.add(fill, inner, outline);
  group.traverse((object) => {
    object.renderOrder = 5; // after the splats, before POI labels
  });
  group.visible = false;
  return group;
}

/** Admin: a toggle that draws the navmesh over the splat, to check it against walls and furniture. */
export function mountNavmeshOverlay(viewer: SplatViewer, navmesh: Navmesh | null): void {
  const panel = adminPanel('Navmesh');
  if (!navmesh) return adminNote(panel, 'This scene has no navmesh yet. Set the scale, then run atomspace navmesh.');

  const lines = navmeshLines(navmesh);
  const group = overlayGroup(lines);
  viewer.scene.add(group);

  const toggle = document.createElement('button');
  toggle.setAttribute('aria-pressed', 'false');
  const summary = document.createElement('p');
  summary.textContent =
    `${navmesh.polygons.length} polygons, ${lines.areaM2.toFixed(1)} m² walkable, ` +
    `keeps ${navmesh.agent.radius_m} m from obstacles. Yellow is the edge.`;
  panel.append(toggle, summary);

  const refresh = () => {
    toggle.textContent = group.visible ? 'Hide navmesh' : 'Show navmesh';
    toggle.setAttribute('aria-pressed', String(group.visible));
  };
  toggle.addEventListener('click', () => {
    group.visible = !group.visible;
    refresh();
  });
  refresh();
}
