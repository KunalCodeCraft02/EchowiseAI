// Self-contained interactive 3D heart component. Pure Three.js (ES modules, no bundler) —
// the vendored copies live in ./vendor/three/, resolved via the "three" import-map entry
// declared on the page that loads this module. Purely presentational: nothing here reads or
// writes application state beyond the one auth check the CTA button performs via window.API.

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';

// ---------------------------------------------------------------------------
// Anatomical pin definitions -- local-space coordinates on the heart group.
// Tuned by eye against the procedural mesh built below, not measured from a real heart.
// ---------------------------------------------------------------------------
// `pos` anchors the glowing dot to the mesh surface; `offset` is a fixed pixel nudge (in
// screen space) that pushes the glass label card clear of the heart so it never overlaps it,
// with a thin connector line drawn from dot to card.
const PIN_DEFS = [
  { id: 'lv',  label: 'Left Ventricle (LVEF)',        pos: [-0.32, -0.46,  0.38], target: 'ventricle',    offset: [-118,  66] },
  { id: 'mv',  label: 'Mitral Valve',                  pos: [-0.14, -0.06,  0.62], target: 'ventricle',    offset: [ 128,  18] },
  { id: 'ao',  label: 'Aortic Root / Ascending Aorta', pos: [ 0.10,  1.28,  0.05], target: 'aorta',        offset: [  92, -34] },
  { id: 'ra',  label: 'Right Atrium',                  pos: [ 0.62,  0.62,  0.10], target: 'atriumRight',  offset: [ 132, -38] },
  { id: 'ivs', label: 'Interventricular Septum',       pos: [ 0.00, -0.10,  0.48], target: 'septum',       offset: [-150,  -6] },
];

const CORE_GLOW = 0xff3366;

function beatCurve(t) {
  // Two-bump "lub-dub" contraction curve over a 0..1 cycle: a sharp systolic contraction
  // (S1) followed by a softer second bump (S2, the dicrotic notch echo), then a slow
  // diastolic relaxation back to baseline. Pure math, no easing library needed.
  const s1 = Math.exp(-Math.pow((t - 0.08) / 0.055, 2));
  const s2 = Math.exp(-Math.pow((t - 0.24) / 0.09, 2)) * 0.45;
  return s1 + s2;
}

function makeIcosphere(radius, detail) {
  const geo = new THREE.IcosahedronGeometry(radius, detail);
  geo.computeVertexNormals();
  return geo;
}

// Deforms a unit-radius icosphere into a stylised heart lump: a tapered apex at the
// bottom, twin rounded lobes at the top, and a gentle front-back flattening.
function sculptVentricleMass(geo) {
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const n = v.clone().normalize();
    const theta = Math.atan2(n.x, n.z);

    let radiusScale = 1;

    // Twin-lobe bulge across the top (the atrial "shoulders" of the silhouette).
    const topness = THREE.MathUtils.smoothstep(n.y, 0.05, 0.9);
    radiusScale += 0.16 * Math.cos(2 * theta) * topness * 0.6;

    // Apex taper: pull the lower third down into a rounded point.
    const bottomness = THREE.MathUtils.smoothstep(-n.y, 0.15, 0.95);
    radiusScale *= 1 - 0.62 * bottomness;

    // Gentle organic ripple so the surface doesn't read as a perfect sphere.
    radiusScale += 0.02 * Math.sin(theta * 5 + n.y * 6);

    v.copy(n).multiplyScalar(radiusScale);
    // Front-back flatten, slight vertical stretch.
    v.x *= 1.02; v.y *= 1.28; v.z *= 0.86;
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

function sculptAtrium(geo, sign) {
  const pos = geo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    v.x *= 1.0; v.y *= 0.85; v.z *= 0.9;
    // Nudge the lobe slightly outward/up so it reads as attached, not buried.
    v.x += sign * 0.08;
    v.y += 0.05;
    pos.setXYZ(i, v.x, v.y, v.z);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

function buildVessel(points, radius) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
  return new THREE.TubeGeometry(curve, 24, radius, 10, false);
}

function glassMaterial(hex, opacity) {
  return new THREE.MeshPhysicalMaterial({
    color: hex,
    transmission: 0.6,
    roughness: 0.15,
    thickness: 0.9,
    clearcoat: 1.0,
    clearcoatRoughness: 0.12,
    ior: 1.42,
    transparent: true,
    opacity: opacity != null ? opacity : 0.92,
    emissive: new THREE.Color(CORE_GLOW),
    emissiveIntensity: 0.06,
    side: THREE.DoubleSide,
  });
}

/**
 * Mounts the interactive 3D heart into `containerEl` (which must be position:relative and
 * sized by CSS). Returns a `{ dispose() }` handle so the caller can tear it down.
 */
export function initHeart3D(containerEl) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeaf0f8);

  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 50);
  camera.position.set(0, 0.15, 4.9);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  containerEl.appendChild(renderer.domElement);
  renderer.domElement.style.cssText = 'position:absolute; inset:0; width:100%; height:100%; display:block; cursor:grab;';

  // ---- Lighting ----
  scene.add(new THREE.AmbientLight(0x9fb3e6, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(2.5, 3, 3);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x6fe3c9, 0.55);
  rim.position.set(-3, 1.5, -2);
  scene.add(rim);
  const coreLight = new THREE.PointLight(CORE_GLOW, 2.2, 4.5, 2);
  coreLight.position.set(0, -0.2, 0.2);
  scene.add(coreLight);

  // ---- Heart group ----
  const heart = new THREE.Group();
  scene.add(heart);

  const parts = {};

  parts.ventricle = new THREE.Mesh(
    sculptVentricleMass(makeIcosphere(1, 4)),
    glassMaterial(0xff4d6d, 0.9),
  );
  heart.add(parts.ventricle);

  parts.atriumLeft = new THREE.Mesh(
    sculptAtrium(makeIcosphere(0.42, 3), -1),
    glassMaterial(0xff6b81, 0.88),
  );
  parts.atriumLeft.position.set(-0.5, 0.85, -0.15);
  heart.add(parts.atriumLeft);

  parts.atriumRight = new THREE.Mesh(
    sculptAtrium(makeIcosphere(0.42, 3), 1),
    glassMaterial(0xff6b81, 0.88),
  );
  parts.atriumRight.position.set(0.55, 0.75, 0.05);
  heart.add(parts.atriumRight);

  parts.aorta = new THREE.Mesh(
    buildVessel([[0, 0.9, 0.1], [0.05, 1.25, 0.15], [0.35, 1.4, 0.05], [0.55, 1.25, -0.25]], 0.16),
    glassMaterial(0xffb3a0, 0.9),
  );
  heart.add(parts.aorta);

  parts.pulmonaryArtery = new THREE.Mesh(
    buildVessel([[-0.15, 0.95, -0.05], [-0.3, 1.2, -0.2], [-0.5, 1.28, -0.45]], 0.12),
    glassMaterial(0xa0c4ff, 0.88),
  );
  heart.add(parts.pulmonaryArtery);

  // A thin internal wall bisecting the ventricular mass front-to-back -- kept just inside the
  // outer surface (rather than spanning its full width) and low-opacity, so it reads as a
  // glimpse of internal structure through the glass rather than a shape poking through it.
  parts.septum = new THREE.Mesh(
    new THREE.BoxGeometry(0.04, 0.68, 0.34, 1, 8, 8),
    new THREE.MeshPhysicalMaterial({
      color: 0xffcfc4, transmission: 0.1, roughness: 0.35, opacity: 0.32,
      transparent: true, side: THREE.DoubleSide, emissive: new THREE.Color(CORE_GLOW), emissiveIntensity: 0.1,
    }),
  );
  parts.septum.position.set(0, -0.1, 0);
  heart.add(parts.septum);

  // Inner glow core -- additive-blended sphere that pulses with the beat.
  const glowCore = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 24, 24),
    new THREE.MeshBasicMaterial({ color: CORE_GLOW, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending }),
  );
  glowCore.position.set(0, -0.25, 0.05);
  heart.add(glowCore);

  heart.rotation.y = -0.35;
  heart.position.y = -0.15;

  const highlightState = {};
  Object.keys(parts).forEach(k => { highlightState[k] = 0; });

  // ---- Controls ----
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, -0.1, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 2.6;
  controls.maxDistance = 6.5;
  controls.enablePan = true;
  controls.panSpeed = 0.4;
  controls.screenSpacePanning = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.9;
  controls.update();

  let idleTimer = null;
  const resumeIdleRotate = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => { controls.autoRotate = true; }, 2200);
  };
  controls.addEventListener('start', () => {
    controls.autoRotate = false;
    renderer.domElement.style.cursor = 'grabbing';
  });
  controls.addEventListener('end', () => {
    renderer.domElement.style.cursor = 'grab';
    resumeIdleRotate();
  });

  // ---- Pin overlay (DOM glass micro-cards + SVG connector lines) ----
  const overlay = document.createElement('div');
  overlay.className = 'heart-pin-overlay';
  containerEl.appendChild(overlay);

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'heart-pin-lines');
  overlay.appendChild(svg);

  const pinEls = PIN_DEFS.map(def => {
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('class', 'heart-pin-line');
    svg.appendChild(line);

    const dot = document.createElement('span');
    dot.className = 'heart-pin-dot';
    overlay.appendChild(dot);

    const el = document.createElement('button');
    el.type = 'button';
    el.className = 'heart-pin';
    el.setAttribute('aria-label', def.label);
    el.innerHTML = `<span class="pin-label">${def.label}</span>`;
    overlay.appendChild(el);

    const setActive = on => {
      el.classList.toggle('active', on);
      dot.classList.toggle('active', on);
      highlightState[def.target] = on ? 1 : 0;
    };
    el.addEventListener('mouseenter', () => setActive(true));
    el.addEventListener('mouseleave', () => setActive(false));
    el.addEventListener('focus', () => setActive(true));
    el.addEventListener('blur', () => setActive(false));
    el.addEventListener('click', () => setActive(!el.classList.contains('active')));

    return { def, el, dot, line, vec: new THREE.Vector3(...def.pos) };
  });

  // ---- Resize handling ----
  function resize() {
    const w = containerEl.clientWidth || 1;
    const h = containerEl.clientHeight || 1;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  const ro = new ResizeObserver(resize);
  ro.observe(containerEl);
  resize();

  // ---- Animation loop ----
  const clock = new THREE.Clock();
  const BEAT_PERIOD = 0.95; // seconds per heartbeat cycle
  let raf = null;

  function updatePins() {
    const rect = containerEl.getBoundingClientRect();
    // Offsets are tuned in pixels against the ~560px reference stage width; on a narrower
    // phone-sized stage they're scaled down proportionally so label cards stay inside the
    // container instead of clipping against its edge.
    const offsetScale = Math.max(0.55, Math.min(1, rect.width / 560));
    for (const p of pinEls) {
      const world = p.vec.clone().applyMatrix4(heart.matrixWorld);
      const ndc = world.clone().project(camera);
      const anchorX = (ndc.x * 0.5 + 0.5) * rect.width;
      const anchorY = (-ndc.y * 0.5 + 0.5) * rect.height;
      const behind = ndc.z > 1;
      const [dx, dy] = p.def.offset;
      const cardX = anchorX + dx * offsetScale;
      const cardY = anchorY + dy * offsetScale;

      p.dot.style.transform = `translate(${anchorX}px, ${anchorY}px) translate(-50%, -50%)`;
      p.el.style.transform = `translate(${cardX}px, ${cardY}px) translate(-50%, -50%)`;
      const vis = behind ? '0' : '1';
      p.dot.style.opacity = vis;
      p.el.style.opacity = vis;
      p.el.style.pointerEvents = behind ? 'none' : 'auto';

      p.line.setAttribute('x1', anchorX);
      p.line.setAttribute('y1', anchorY);
      p.line.setAttribute('x2', cardX);
      p.line.setAttribute('y2', cardY);
      p.line.style.opacity = behind ? '0' : '0.55';
    }
  }

  function animate() {
    raf = requestAnimationFrame(animate);
    const t = clock.getElapsedTime();
    const cyclePos = (t % BEAT_PERIOD) / BEAT_PERIOD;
    const beat = beatCurve(cyclePos);

    heart.scale.set(1 + beat * 0.045, 1 + beat * 0.07, 1 + beat * 0.045);
    glowCore.material.opacity = 0.28 + beat * 0.4;
    glowCore.scale.setScalar(1 + beat * 0.5);
    coreLight.intensity = 1.6 + beat * 2.4;

    for (const key of Object.keys(parts)) {
      const mat = parts[key].material;
      const targetGlow = 0.06 + highlightState[key] * 0.55;
      mat.emissiveIntensity += (targetGlow - mat.emissiveIntensity) * 0.15;
    }

    controls.update();
    // Keep pan bounded near the origin so the heart never drifts off-centre.
    if (controls.target.length() > 0.6) controls.target.setLength(0.6);

    renderer.render(scene, camera);
    updatePins();
  }
  animate();

  function dispose() {
    cancelAnimationFrame(raf);
    clearTimeout(idleTimer);
    ro.disconnect();
    controls.dispose();
    renderer.dispose();
    overlay.remove();
    renderer.domElement.remove();
    Object.values(parts).forEach(m => { m.geometry.dispose(); m.material.dispose(); });
    glowCore.geometry.dispose();
    glowCore.material.dispose();
  }

  return { dispose };
}
