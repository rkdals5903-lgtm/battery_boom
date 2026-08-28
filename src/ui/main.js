import * as THREE from 'three';

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

function roundedRectShape(w, h, r) {
  const x = -w / 2;
  const y = -h / 2;
  const s = new THREE.Shape();
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

function roundedExtrude(w, h, d, r = 0.18, bevel = 0.02) {
  return new THREE.ExtrudeGeometry(roundedRectShape(w, h, r), {
    depth: d,
    bevelEnabled: true,
    bevelSize: bevel,
    bevelThickness: bevel,
    bevelSegments: 3,
    curveSegments: 24,
  }).rotateX(Math.PI / 2).translate(0, d / 2, 0);
}

function makeTextSprite(text, color = '#ffffff', scale = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = '700 112px Arial';
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2.8 * scale, .7 * scale, 1);
  return sprite;
}

function buildBatteryPack(scale = 1) {
  const g = new THREE.Group();

  const shellMat = new THREE.MeshPhysicalMaterial({
    color: 0x353d45,
    roughness: .28,
    metalness: .26,
    clearcoat: .42,
    clearcoatRoughness: .10,
  });
  const trimMat = new THREE.MeshStandardMaterial({ color: 0x1d232a, roughness: .58, metalness: .34 });
  const innerMat = new THREE.MeshPhysicalMaterial({ color: 0x44515d, roughness: .26, metalness: .12, clearcoat: .18 });
  const metalMat = new THREE.MeshStandardMaterial({ color: 0x96a4b3, roughness: .28, metalness: .72 });

  const lower = new THREE.Group();
  g.add(lower);

  const lowerShell = new THREE.Mesh(roundedExtrude(4.6, 3.0, 0.95, 0.34, 0.03), shellMat);
  lowerShell.position.y = 0.48;
  lowerShell.castShadow = true;
  lowerShell.receiveShadow = true;
  lower.add(lowerShell);

  const lip = new THREE.Mesh(roundedExtrude(4.35, 2.75, 0.10, 0.28, 0.02), trimMat);
  lip.position.y = 0.93;
  lower.add(lip);

  const floor = new THREE.Mesh(roundedExtrude(3.8, 2.2, 0.04, 0.2, 0.01), new THREE.MeshStandardMaterial({ color: 0x242a30, roughness: .96 }));
  floor.position.y = 0.18;
  lower.add(floor);

  const wallXGeo = new THREE.BoxGeometry(3.82, 1.00, 0.12);
  const wallZGeo = new THREE.BoxGeometry(0.12, 1.00, 2.08);
  const fWall = new THREE.Mesh(wallXGeo, innerMat);
  fWall.position.set(0, 0.58, 1.03);
  const bWall = fWall.clone(); bWall.position.z = -1.03;
  const lWall = new THREE.Mesh(wallZGeo, innerMat); lWall.position.set(-1.91, 0.58, 0);
  const rWall = lWall.clone(); rWall.position.x = 1.91;
  lower.add(fWall, bWall, lWall, rWall);

  for (const x of [-2.02, 2.02]) {
    const block = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.28, 0.70), trimMat);
    block.position.set(x, 0.70, 0);
    lower.add(block);
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.04, 10, 28, Math.PI), metalMat);
    handle.rotation.z = Math.PI / 2;
    handle.rotation.y = x < 0 ? Math.PI / 2 : -Math.PI / 2;
    handle.position.set(x + (x < 0 ? -0.03 : 0.03), 0.76, 0);
    lower.add(handle);
  }

  const warning = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.02, 0.28), new THREE.MeshStandardMaterial({ color: 0xf2ad36, roughness: .72 }));
  warning.position.set(0, 0.72, 1.13);
  warning.rotation.x = Math.PI / 2;
  lower.add(warning);

  const lidPivot = new THREE.Group();
  lidPivot.position.set(0, 1.0, -1.03);
  g.add(lidPivot);

  const lidGroup = new THREE.Group();
  lidGroup.position.z = 1.03;
  lidPivot.add(lidGroup);

  const lid = new THREE.Mesh(roundedExtrude(4.6, 3.0, 0.22, 0.34, 0.02), new THREE.MeshPhysicalMaterial({ color: 0x5a6470, roughness: .18, metalness: .10, clearcoat: .86, clearcoatRoughness: .05 }));
  lid.position.y = 0.08;
  lid.castShadow = true;
  lidGroup.add(lid);

  const inset = new THREE.Mesh(roundedExtrude(4.0, 2.42, 0.04, 0.23, 0.01), new THREE.MeshPhysicalMaterial({ color: 0x718090, roughness: .12, metalness: .08, clearcoat: 1.0, clearcoatRoughness: .02 }));
  inset.position.y = 0.19;
  lidGroup.add(inset);

  const boltPositions = [];
  for (const x of [-1.45, 1.45]) {
    for (const z of [-.92, .92]) {
      boltPositions.push(new THREE.Vector3(x, 0.25, z));
      const bolt = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.03, 12), metalMat);
      bolt.position.set(x, 0.25, z);
      lidGroup.add(bolt);
    }
  }

  g.scale.setScalar(scale);
  g.userData = { lidPivot, lidGroup, boltPositions };
  return g;
}

function setBatteryOpen(pack, t) {
  const p = pack.userData.lidPivot;
  const lidGroup = pack.userData.lidGroup;
  if (!p || !lidGroup) return;
  const s = clamp(t, 0, 1);
  const smooth = s * s * (3 - 2 * s);
  const lift = Math.sin(smooth * Math.PI) * 0.08 + smooth * 0.10;
  p.rotation.x = -2.05 * smooth;
  lidGroup.position.z = 1.03 - 0.14 * smooth;
  lidGroup.position.y = 0.02 + lift;
  lidGroup.rotation.z = -0.03 * Math.sin(smooth * Math.PI);
}

function createCell() {
  const g = new THREE.Group();
  const green = new THREE.MeshPhysicalMaterial({
    color: 0x3eb95e,
    roughness: .28,
    metalness: .03,
    clearcoat: .34,
    clearcoatRoughness: .10,
  });
  const greenDark = new THREE.MeshStandardMaterial({
    color: 0x227644,
    roughness: .45,
    metalness: .03,
  });
  const terminal = new THREE.MeshStandardMaterial({
    color: 0xdce4e9,
    roughness: .26,
    metalness: .58,
  });

  const body = new THREE.Mesh(new THREE.BoxGeometry(.62, 1.12, .38), green);
  body.position.y = .56;
  body.castShadow = true;
  body.receiveShadow = true;
  g.add(body);

  const topPlate = new THREE.Mesh(new THREE.BoxGeometry(.64, .07, .40), greenDark);
  topPlate.position.y = 1.155;
  g.add(topPlate);

  const terminalA = new THREE.Mesh(new THREE.BoxGeometry(.11, .055, .08), terminal);
  terminalA.position.set(-.18, 1.225, 0);
  const terminalB = terminalA.clone();
  terminalB.position.x = .18;
  g.add(terminalA, terminalB);

  return g;
}

function buildScrewdriver() {
  const g = new THREE.Group();
  const metal = new THREE.MeshStandardMaterial({ color: 0xb9c2cb, roughness: .28, metalness: .7 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2b3138, roughness: .42, metalness: .25 });

  const head = new THREE.Mesh(new THREE.CylinderGeometry(.28, .28, .55, 22), dark);
  head.position.y = .28;
  g.add(head);

  const collar = new THREE.Mesh(new THREE.CylinderGeometry(.16, .16, .16, 16), metal);
  collar.position.y = -.03;
  g.add(collar);

  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(.05, .05, .90, 14), metal);
  shaft.position.y = -.56;
  g.add(shaft);

  const bit = new THREE.Mesh(new THREE.ConeGeometry(.06, .20, 4), metal);
  bit.rotation.y = Math.PI / 4;
  bit.position.y = -1.11;
  g.add(bit);

  return g;
}

class StoryUI {
  constructor() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x09111a);
    this.scene.fog = new THREE.Fog(0x09111a, 16, 42);

    this.camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    $('scene').appendChild(this.renderer.domElement);

    this.clock = new THREE.Clock();
    this.mode = 'menu';
    this.transition = null;
    this.openedCount = 0;
    this.rebuiltCount = 0;
    this.goodPool = 0;
    this.buildCaseCount = 0;
    this.cycle = null;
    this.currentCellVisual = null;
    this.cellShowProgress = 0;
    this.beltOffset = 0;
    this.fadeEl = $('transitionFade');
    this.liveStatus = null;
    this.lastStatusSignature = '';
    this.resultHoldUntil = 0;
    this.motionHoldUntil = 0;
    this.pendingStatus = null;
    this.expiredResultSignature = null;
    this.backendResetInProgress = false;
    this.livePhaseElapsed = 0;
    this.setFade(0);

    this.menuCam = { pos: new THREE.Vector3(8.2, 6.2, 9.2), target: new THREE.Vector3(0, 1.3, 0) };
    this.zoomCam = { pos: new THREE.Vector3(1.0, 2.2, 2.3), target: new THREE.Vector3(0, 0.8, 0.2) };
    this.inspectCam = { pos: new THREE.Vector3(0, 3.2, 8.8), target: new THREE.Vector3(0, 1.6, 0) };
    this.camCurrentTarget = this.menuCam.target.clone();

    this.buildScene();
    this.bindUI();
    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.startBackendSync();
    this.animate();
  }

  buildScene() {
    const hemi = new THREE.HemisphereLight(0xdce7f2, 0x0a1016, 1.15);
    this.scene.add(hemi);

    this.key = new THREE.DirectionalLight(0xf6fbff, 2.7);
    this.key.position.set(-8, 11, 8);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(2048, 2048);
    this.key.shadow.camera.left = -12;
    this.key.shadow.camera.right = 12;
    this.key.shadow.camera.top = 12;
    this.key.shadow.camera.bottom = -12;
    this.scene.add(this.key);

    this.rim = new THREE.DirectionalLight(0x4ba7ff, 1.6);
    this.rim.position.set(6, 4, -4);
    this.scene.add(this.rim);

    this.floor = new THREE.Mesh(new THREE.CircleGeometry(11.8, 72), new THREE.MeshStandardMaterial({ color: 0x1a2431, roughness: .96, metalness: .03 }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.scene.add(this.floor);

    this.halo = new THREE.Mesh(new THREE.CircleGeometry(7.0, 56), new THREE.MeshBasicMaterial({ color: 0x59aaff, transparent: true, opacity: .08, depthWrite: false }));
    this.halo.rotation.x = -Math.PI / 2;
    this.halo.position.y = 0.01;
    this.scene.add(this.halo);

    this.menuBattery = buildBatteryPack(1.6);
    this.menuBattery.position.y = 0.2;
    this.scene.add(this.menuBattery);

    this.inspectWorld = new THREE.Group();
    this.inspectWorld.visible = false;
    this.scene.add(this.inspectWorld);

    this.backgroundPanel = new THREE.Mesh(new THREE.PlaneGeometry(32, 18), new THREE.MeshBasicMaterial({ color: 0x0d1721 }));
    this.backgroundPanel.position.set(0, 6.2, -12);
    this.inspectWorld.add(this.backgroundPanel);

    this.labPanel = new THREE.Mesh(new THREE.PlaneGeometry(32, 18), new THREE.MeshBasicMaterial({ color: 0x17283b, transparent: true, opacity: 0.05 }));
    this.labPanel.position.set(0, 6.2, -11.9);
    this.inspectWorld.add(this.labPanel);

    this.insideGlow = new THREE.PointLight(0xb8e1ff, 0.0, 24, 1.7);
    this.insideGlow.position.set(0, 2.8, 0);
    this.inspectWorld.add(this.insideGlow);

    // Layout positions
    this.pos = {
      palletIn: new THREE.Vector3(-5.4, 1.10, -1.7),
      beltStart: new THREE.Vector3(-6.4, 1.10, 0),
      table: new THREE.Vector3(0, 1.10, 0),
      discard: new THREE.Vector3(4.8, 1.10, -1.6),
      beltOut: new THREE.Vector3(4.3, 1.10, 0),
      palletOut: new THREE.Vector3(5.6, 1.10, 1.6),
    };

    // Conveyor lane
    this.conveyorGroup = new THREE.Group();
    this.inspectWorld.add(this.conveyorGroup);
    const beltBase = new THREE.Mesh(new THREE.BoxGeometry(10.8, 0.20, 2.0), new THREE.MeshStandardMaterial({ color: 0x434a52, roughness: .90 }));
    beltBase.position.y = 0.68;
    beltBase.receiveShadow = true;
    this.conveyorGroup.add(beltBase);
    const railMat = new THREE.MeshStandardMaterial({ color: 0x4787cc, roughness: .4, metalness: .18 });
    for (const z of [-1.0, 1.0]) {
      const rail = new THREE.Mesh(new THREE.BoxGeometry(10.9, 0.08, 0.08), railMat);
      rail.position.set(0, 0.88, z * 0.8);
      this.conveyorGroup.add(rail);
    }
    this.beltSlats = [];
    for (let i = 0; i < 22; i++) {
      const slat = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.03, 1.56), new THREE.MeshStandardMaterial({ color: 0x5a6168, roughness: .82 }));
      slat.position.set(-5.2 + i * 0.50, 0.80, 0);
      this.conveyorGroup.add(slat);
      this.beltSlats.push(slat);
    }

    // Table
    this.inspectPlatform = new THREE.Mesh(new THREE.BoxGeometry(5.8, 0.35, 4.2), new THREE.MeshStandardMaterial({ color: 0xd7e0e9, roughness: .5, metalness: .05, transparent: true, opacity: 0.0 }));
    this.inspectPlatform.position.set(0, 0.50, 0);
    this.inspectPlatform.receiveShadow = true;
    this.inspectWorld.add(this.inspectPlatform);

    // Pallets and discard bin
    const palletMat = new THREE.MeshStandardMaterial({ color: 0x9c7b55, roughness: .8 });
    this.palletIn = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.18, 1.8), palletMat);
    this.palletIn.position.set(this.pos.palletIn.x, 0.78, this.pos.palletIn.z);
    this.inspectWorld.add(this.palletIn);
    this.palletOut = this.palletIn.clone();
    this.palletOut.position.set(this.pos.palletOut.x, 0.78, this.pos.palletOut.z);
    this.inspectWorld.add(this.palletOut);
    this.palletInHome = this.palletIn.position.clone();
    this.palletInOff = this.palletIn.position.clone().add(new THREE.Vector3(-6.0, 0, 0));
    this.palletOutHome = this.palletOut.position.clone();
    this.palletOutOff = this.palletOut.position.clone().add(new THREE.Vector3(3.2, 0, 0));
    this.palletOut.position.copy(this.palletOutOff);

    // Batteries
    this.inspectBattery = buildBatteryPack(1.0);
    this.inspectBattery.position.copy(this.pos.palletIn);
    this.inspectWorld.add(this.inspectBattery);

    this.rebuiltBattery = buildBatteryPack(1.0);
    this.rebuiltBattery.visible = false;
    this.rebuiltBattery.position.copy(this.pos.table);
    this.inspectWorld.add(this.rebuiltBattery);

    // Screwdriver
    this.driver = buildScrewdriver();
    this.driver.visible = false;
    this.inspectWorld.add(this.driver);

    // Cell inspection visual area
    this.cellPedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.72, 0.86, 0.22, 28), new THREE.MeshStandardMaterial({ color: 0xe7eff6, roughness: .38, metalness: .05 }));
    this.cellPedestal.position.set(0, 1.12, 3.0);
    this.cellPedestal.visible = false;
    this.inspectWorld.add(this.cellPedestal);

    this.cellBackGlow = new THREE.Mesh(new THREE.CircleGeometry(0.86, 32), new THREE.MeshBasicMaterial({ color: 0x7ec5ff, transparent: true, opacity: 0.0, depthWrite: false }));
    this.cellBackGlow.position.set(0, 1.64, 2.65);
    this.inspectWorld.add(this.cellBackGlow);

    this.rebuildCells = [];
    for (let i = 0; i < 4; i++) {
      const c = createCell();
      c.visible = false;
      c.scale.setScalar(0.48);
      this.inspectWorld.add(c);
      this.rebuildCells.push(c);
    }

    this.camera.position.copy(this.menuCam.pos);
    this.camera.lookAt(this.menuCam.target);
  }

  bindUI() {
    $('openBtn').addEventListener('click', () => this.openSequence());
    $('exitBtn').addEventListener('click', () => this.exitSequence());
  }

  startBackendSync() {
    const poll = async () => {
      try {
        const response = await fetch('/api/status', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.applyBackendStatus(await response.json());
      } catch (error) {
        const badge = $('connectionBadge');
        badge.className = 'connection-badge offline';
        badge.textContent = 'UI 연동 서버 연결 끊김';
      }
    };
    poll();
    this.statusPollTimer = window.setInterval(poll, 700);
  }

  applyBackendStatus(status) {
    const now = performance.now();

    // main.py/프로젝트가 종료되어 공정 노드가 ROS graph에서 사라지면
    // UI도 자동으로 초기 화면으로 복귀한다.
    if (status.projectRunning === false) {
      this.liveStatus = status;
      $('cellInfoCard').classList.add('hidden');
      if (this.mode === 'inspection' && !this.backendResetInProgress) {
        this.backendResetInProgress = true;
        this.pendingStatus = null;
        this.resultHoldUntil = 0;
        this.motionHoldUntil = 0;
        this.exitSequence();
      }
      return;
    }
    this.backendResetInProgress = false;

    const signature = [
      status.phase,
      status.cellIndex,
      status.voltageOk,
      status.cnnOk,
      status.judgement,
      status.goodCellPool,
      status.rebuiltCount
    ].join('|');

    const activeHoldUntil = Math.max(this.resultHoldUntil, this.motionHoldUntil);
    const newCellMeasurement = Boolean(
      status.phase === 'cell_sequence' &&
      status.cellIndex &&
      !status.judgement &&
      this.liveStatus &&
      Number(status.cellIndex) !== Number(this.liveStatus.cellIndex)
    );

    // 이전 셀 결과를 2초 유지하는 중이라도 다음 셀 측정이 실제로 시작되면
    // 이전 결과를 즉시 지우고 새 셀의 '측정 중' 화면으로 전환한다.
    if (newCellMeasurement) {
      this.resultHoldUntil = 0;
      this.pendingStatus = null;
    } else if (
      activeHoldUntil > now &&
      signature !== this.lastStatusSignature
    ) {
      this.pendingStatus = status;
      return;
    }

    if (signature !== this.lastStatusSignature) {
      this.lastStatusSignature = signature;
      this.livePhaseElapsed = 0;
      this.setupLivePhase(status);

      // 최종 검사 결과는 5초간 고정.
      this.resultHoldUntil =
        status.phase === 'cell_sequence' && status.judgement
          ? now + 5000
          : 0;

      // 실제 로그가 다음 단계로 빨리 넘어가도 뚜껑 개폐 모션은 끝까지 천천히 보여준다.
      if (status.phase === 'opening_battery') {
        this.motionHoldUntil = now + 5200;
      } else if (status.phase === 'close_new_battery') {
        this.motionHoldUntil = now + 5200;
      } else {
        this.motionHoldUntil = 0;
      }
    }

    this.liveStatus = status;

    // 최종 결과 5초 유지가 끝난 뒤 같은 셀의 동일한 FINAL 상태가
    // API polling으로 반복되어도 결과 카드를 다시 띄우지 않는다.
    if (
      status.phase === 'cell_sequence' &&
      status.judgement &&
      this.resultHoldUntil > 0 &&
      now >= this.resultHoldUntil
    ) {
      this.expiredResultSignature = signature;
      this.resultHoldUntil = 0;
      $('cellInfoCard').classList.add('hidden');
    }

    const badge = $('connectionBadge');
    badge.className = `connection-badge ${status.connected ? '' : 'offline'}`.trim();
    badge.textContent = status.connected
      ? (status.source === 'network' ? 'ROS2 원격 로그 연결됨' : 'ROS2 로컬 로그 연결됨')
      : 'ROS2 원격 로그 연결 대기';
    this.openedCount = Number(status.openedCount || 0);
    this.rebuiltCount = Number(status.rebuiltCount || 0);
    this.goodPool = Number(status.goodCellPool || 0);
    this.buildCaseCount = this.goodPool;
    this.setCounts();
    this.setPhaseText(status.error ? `${status.phaseText} · 오류 감지` : status.phaseText);

    if (status.phase !== 'cell_sequence' || !status.cellIndex) {
      $('cellInfoCard').classList.add('hidden');
      return;
    }

    if (status.judgement && this.expiredResultSignature === signature) {
      $('cellInfoCard').classList.add('hidden');
      return;
    }

    // 새 셀이 측정 상태로 들어오면 이전 셀 결과 만료 표시를 해제한다.
    if (!status.judgement) {
      this.expiredResultSignature = null;
    }

    $('batteryOrder').textContent = `배터리 #${Math.max(1, status.batteryOrder || 1)}`;
    $('cellOrder').textContent = `CELL ${status.cellIndex} / 4`;

    const threshold = status.threshold == null ? 10 : Number(status.threshold);
    const finalReady = Boolean(status.judgement);

    if (finalReady) {
      $('cellVoltage').textContent = status.voltage == null ? '-' : `${Number(status.voltage).toFixed(3)} V`;
      $('voltageHint').textContent = `판정 기준 ${threshold.toFixed(2)} V 이상`;

      const cnnKnown = typeof status.cnnOk === 'boolean';
      const cnnGood = status.cnnOk === true;
      $('cellAppearance').textContent = cnnKnown ? (cnnGood ? '정상' : '불량') : '-';
      $('appearanceBadge').className = `status-badge small ${cnnKnown ? (cnnGood ? 'good' : 'bad') : 'neutral'}`;
      $('appearanceBadge').textContent = cnnKnown ? (cnnGood ? 'CNN 정상' : 'CNN 불량') : 'CNN 검사 대기';

      const good = status.judgement === 'pass';
      const judgement = $('judgementBadge');
      judgement.className = `status-badge ${good ? 'good' : 'bad'}`;
      judgement.textContent = good ? '양품' : '불량';
      if (good) {
        $('cellJudgement').textContent = '전압 + CNN 최종 합격 / 재사용 가능';
      } else if (status.voltageOk === false && status.cnnOk === true) {
        $('cellJudgement').textContent = 'CNN 정상 · 전압 불합격 → 불량';
      } else if (status.voltageOk === true && status.cnnOk === false) {
        $('cellJudgement').textContent = '전압 정상 · CNN 불량 → 불량';
      } else {
        $('cellJudgement').textContent = '최종 불합격 / 폐기 대상';
      }
    } else {
      // 검사 완료 전에는 이전 셀의 수치/판정이 절대 남지 않도록 측정 상태만 표시한다.
      $('cellVoltage').textContent = '측정 중';
      $('voltageHint').textContent = '전압 측정 중';
      $('cellAppearance').textContent = '판정 중';
      $('appearanceBadge').className = 'status-badge neutral small';
      $('appearanceBadge').textContent = 'CNN 판정 중';
      const judgement = $('judgementBadge');
      judgement.className = 'status-badge neutral';
      judgement.textContent = '검사 중';
      $('cellJudgement').textContent = '전압 및 CNN 결과 측정 중';
    }

    $('cellInfoCard').classList.remove('hidden');
  }

  prepareLiveView() {
    this.cycle = null;
    this.inspectBattery.visible = false;
    this.inspectBattery.position.copy(this.pos.palletIn);
    this.inspectBattery.rotation.set(0, 0, 0);
    setBatteryOpen(this.inspectBattery, 0);

    this.rebuiltBattery.visible = false;

    // 실제 공정에서 필요할 때만 등장하도록 기본 상태는 모두 숨긴다.
    this.inspectPlatform.visible = false;
    this.inspectPlatform.material.opacity = 0;
    this.inspectPlatform.position.y = -0.75;

    this.palletIn.visible = false;
    this.palletIn.position.copy(this.palletInOff);
    this.palletOut.visible = false;
    this.palletOut.position.copy(this.palletOutOff);

    this.driver.visible = false;
    this.cellPedestal.visible = false;
    this.labPanel.material.opacity = .08;

    this.setCounts();
    if (this.liveStatus) {
      this.setupLivePhase(this.liveStatus);
      this.applyBackendStatus(this.liveStatus);
    }
  }

  setupLivePhase(status) {
    if (!this.inspectBattery || !this.rebuiltBattery) return;

    $('cellInfoCard').classList.add('hidden');
    this.driver.visible = false;
    this.cellPedestal.visible = false;
    this.cellBackGlow.material.opacity = 0;
    this.removeCurrentCell();

    // 공정 전환 때 이전 장비가 남지 않도록 일단 모두 숨긴 뒤 필요한 것만 켠다.
    this.inspectPlatform.visible = false;
    this.inspectPlatform.material.opacity = 0;
    this.inspectPlatform.position.y = -0.75;

    this.palletIn.visible = false;
    this.palletIn.position.copy(this.palletInOff);
    this.palletOut.visible = false;
    this.palletOut.position.copy(this.palletOutOff);

    this.inspectBattery.visible = false;
    this.rebuiltBattery.visible = false;

    const phase = status.phase;

    if (phase === 'pallet_to_conveyor') {
      this.palletIn.visible = true;
      this.palletIn.position.copy(this.palletInHome);
      this.inspectBattery.visible = true;
      this.inspectBattery.position.copy(this.pos.palletIn);
      this.inspectBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.inspectBattery, 0);

    } else if (phase === 'conveyor_to_table') {
      this.inspectBattery.visible = true;
      this.inspectBattery.position.copy(this.pos.beltStart);
      this.inspectBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.inspectBattery, 0);

    } else if (['unscrew', 'opening_battery', 'cell_sequence'].includes(phase)) {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.inspectBattery.visible = true;
      this.inspectBattery.position.copy(this.pos.table);
      this.inspectBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.inspectBattery, phase === 'unscrew' ? 0 : 1);

    } else if (phase === 'discard_motion') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.inspectBattery.visible = true;
      this.inspectBattery.position.copy(this.pos.table);
      this.inspectBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.inspectBattery, 1);

    } else if (['close_new_battery', 'screw_in_new', 'rebuilt_complete'].includes(phase)) {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.table);
      this.rebuiltBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.rebuiltBattery, phase === 'close_new_battery' ? 1 : 0);

    } else if (['new_to_conveyor', 'process_complete'].includes(phase)) {
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.table);
      this.rebuiltBattery.rotation.set(0, 0, 0);
      setBatteryOpen(this.rebuiltBattery, 0);
    }

    if (phase === 'cell_sequence' && status.cellIndex) {
      // 새 셀 시작 시 이전 셀 UI를 먼저 초기화한다.
      $('cellVoltage').textContent = '측정 중';
      $('voltageHint').textContent = '전압 측정 중';
      $('cellAppearance').textContent = '판정 중';
      $('appearanceBadge').className = 'status-badge neutral small';
      $('appearanceBadge').textContent = 'CNN 판정 중';
      $('judgementBadge').className = 'status-badge neutral';
      $('judgementBadge').textContent = '검사 중';
      $('cellJudgement').textContent = '전압 및 CNN 결과 측정 중';

      // CNN은 정상/불량만 알려주므로 셀 형상은 항상 동일한 초록 직육면체로 표시한다.
      this.currentCellVisual = createCell();
      this.currentCellVisual.scale.setScalar(.85);
      this.currentCellVisual.position.set(0, 1.55, 2.75);
      this.inspectWorld.add(this.currentCellVisual);
      this.cellPedestal.visible = true;
    }
  }

  updateLiveVisuals(dt, time) {
    const status = this.liveStatus;
    if (!status) {
      this.updateBelt(dt * .12);
      return;
    }

    this.livePhaseElapsed += dt;
    const t = this.livePhaseElapsed;
    const phase = status.phase;

    // 기본은 느리게, 실제 컨베이어 공정에서는 눈에 보이도록 벨트를 움직인다.
    this.updateBelt(dt * (phase.includes('conveyor') ? .72 : .10));

    if (phase === 'pallet_to_conveyor') {
      this.palletIn.visible = true;
      this.inspectPlatform.visible = false;
      const p = easeInOut(clamp(t / 4.2, 0, 1));
      this.moveBatteryNatural(
        this.inspectBattery,
        this.pos.palletIn,
        this.pos.beltStart,
        p,
        time,
        'pallet_to_belt'
      );
      if (p > .68) {
        const hide = easeInOut(clamp((p - .68) / .32, 0, 1));
        this.palletIn.position.lerpVectors(this.palletInHome, this.palletInOff, hide);
        if (hide > .98) this.palletIn.visible = false;
      }

    } else if (phase === 'conveyor_to_table') {
      this.palletIn.visible = false;
      const p = clamp(t / 6.2, 0, 1);
      const travelP = easeInOut(clamp(p / .72, 0, 1));
      const settleP = easeInOut(clamp((p - .72) / .28, 0, 1));
      const hover = this.pos.table.clone();
      hover.y += .18;

      if (p < .72) {
        this.inspectPlatform.visible = false;
        this.moveBatteryNatural(
          this.inspectBattery,
          this.pos.beltStart,
          hover,
          travelP,
          time,
          'belt'
        );
      } else {
        this.inspectPlatform.visible = true;
        this.inspectPlatform.material.opacity = settleP;
        this.inspectPlatform.position.y = lerp(-.75, .94, settleP);
        this.moveBatteryNatural(
          this.inspectBattery,
          hover,
          this.pos.table,
          settleP,
          time,
          'belt'
        );
        this.inspectBattery.rotation.x = THREE.MathUtils.lerp(this.inspectBattery.rotation.x, 0, .18);
        this.inspectBattery.rotation.y = THREE.MathUtils.lerp(this.inspectBattery.rotation.y, 0, .18);
        this.inspectBattery.rotation.z = THREE.MathUtils.lerp(this.inspectBattery.rotation.z, 0, .18);
      }

      const camP = easeInOut(p);
      this.camera.position.lerpVectors(
        new THREE.Vector3(-3.4, 3.1, 9.4),
        new THREE.Vector3(.2, 3.05, 7.6),
        camP
      );
      this.camCurrentTarget.lerpVectors(
        new THREE.Vector3(-3.1, 1.2, 0),
        new THREE.Vector3(0, 1.35, 0),
        camP
      );
      this.camera.lookAt(this.camCurrentTarget);

    } else if (phase === 'unscrew') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.inspectBattery.rotation.set(0, 0, 0);

      // 실제 로그 단계가 길어져도 드라이버가 멈춰 보이지 않도록 4개 나사 모션 + 짧은 대기를 반복.
      const loop = 8.3;
      const local = t % loop;
      if (local < 6.45) {
        this.animateDriverSequence(this.inspectBattery, local, false);
      } else {
        this.driver.visible = false;
      }

    } else if (phase === 'opening_battery') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      const lidP = easeInOut(clamp(t / 5.0, 0, 1));
      setBatteryOpen(this.inspectBattery, lidP);
      this.insideGlow.intensity = .75 + lidP * 1.15;

    } else if (phase === 'cell_sequence' && this.currentCellVisual) {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;

      const appear = easeInOut(clamp(Math.min(t, 1.1) / 1.1, 0, 1));
      const cell = this.currentCellVisual;
      const from = new THREE.Vector3(0, 1.55, 2.75);
      cell.visible = true;
      cell.scale.setScalar(.35 + appear * .65);
      cell.rotation.y += dt * .55;
      cell.position.copy(from);
      cell.position.y += Math.sin(time * 2.2) * .045;
      this.cellBackGlow.material.opacity = .15 + Math.sin(time * 3.2) * .04;

      // 최종 전압/CNN 판정이 나온 뒤 5초 동안 결과 화면을 유지한다.
      if (status.judgement && t > 5.0) {
        const move = easeInOut(clamp((t - 5.0) / 1.8, 0, 1));

        if (status.judgement === 'pass') {
          const to = this.getHiddenCaseTarget(
            Math.max(0, Number(status.goodCellPool || 1) - 1)
          );
          cell.position.lerpVectors(from, to, move);
          cell.position.y += Math.sin(move * Math.PI) * .58;
          cell.rotation.y += dt * 2.2;
          this.cellBackGlow.material.opacity = .22 + move * .38;
          if (move < .98) {
            this.setPhaseText(`새로운 케이스에 배치 중 · ${status.goodCellPool} / 4`);
          }
        } else {
          // 불량 셀도 형상은 동일하게 유지하고, 판정 결과에 따라 화면 밖으로 이동만 한다.
          const to = new THREE.Vector3(4.8, 1.5, -1.8);
          cell.position.lerpVectors(from, to, move);
          cell.position.y += Math.sin(move * Math.PI) * .34;
          this.cellBackGlow.material.opacity = .18 + move * .18;
        }
      }

    } else if (phase === 'discard_motion') {
      $('cellInfoCard').classList.add('hidden');

      const p = easeInOut(clamp(t / 4.2, 0, 1));
      const start = this.pos.table.clone();
      const end = new THREE.Vector3(
        this.pos.discard.x + 2.2,
        this.pos.discard.y - .15,
        this.pos.discard.z - 1.6
      );

      // 초반에는 작업대 위에 잠시 머무르고, 이후 오른쪽 뒤쪽으로
      // 들렸다가 기울어지며 화면 밖으로 사라지는 폐기 모션.
      const liftP = easeInOut(clamp(p / .28, 0, 1));
      const flyP = easeInOut(clamp((p - .20) / .80, 0, 1));

      this.inspectBattery.visible = true;
      this.inspectBattery.position.lerpVectors(start, end, flyP);
      this.inspectBattery.position.y +=
        Math.sin(flyP * Math.PI) * .85 +
        liftP * .16;
      this.inspectBattery.rotation.x = flyP * .22;
      this.inspectBattery.rotation.y = flyP * .35;
      this.inspectBattery.rotation.z = -flyP * .68;

      const tableHide = easeInOut(clamp((p - .55) / .38, 0, 1));
      this.inspectPlatform.visible = tableHide < .99;
      this.inspectPlatform.material.opacity = 1 - tableHide;
      this.inspectPlatform.position.y = lerp(.94, -.75, tableHide);

      if (p > .98) {
        this.inspectBattery.visible = false;
        this.inspectPlatform.visible = false;
      }

    } else if (phase === 'close_new_battery') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.rebuiltBattery.visible = true;
      const lidP = easeInOut(clamp(t / 5.0, 0, 1));
      setBatteryOpen(this.rebuiltBattery, 1 - lidP);
      this.insideGlow.intensity = 1.15 - lidP * .55;

    } else if (phase === 'screw_in_new') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.rebuiltBattery.visible = true;
      setBatteryOpen(this.rebuiltBattery, 0);
      const loop = 8.3;
      const local = t % loop;
      if (local < 6.45) {
        this.animateDriverSequence(this.rebuiltBattery, local, true);
      } else {
        this.driver.visible = false;
      }

    } else if (phase === 'rebuilt_complete') {
      this.inspectPlatform.visible = true;
      this.inspectPlatform.material.opacity = 1;
      this.inspectPlatform.position.y = .94;
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.table);
      this.rebuiltBattery.position.y += Math.sin(time * 1.6) * .015;

    } else if (phase === 'new_to_conveyor') {
      this.rebuiltBattery.visible = true;

      // 한 단계 안에서 작업대 -> 컨베이어 -> 팔레트 적재까지 이어서 보여준다.
      const p = clamp(t / 7.0, 0, 1);
      const toBelt = easeInOut(clamp(p / .50, 0, 1));
      const toPallet = easeInOut(clamp((p - .50) / .50, 0, 1));

      if (p < .50) {
        this.inspectPlatform.visible = true;
        const tableHide = easeInOut(clamp(p / .42, 0, 1));
        this.inspectPlatform.material.opacity = 1 - tableHide;
        this.inspectPlatform.position.y = lerp(.94, -.75, tableHide);
        this.moveBatteryNatural(
          this.rebuiltBattery,
          this.pos.table,
          this.pos.beltOut,
          toBelt,
          time,
          'belt'
        );
        this.palletOut.visible = false;
      } else {
        this.inspectPlatform.visible = false;
        this.palletOut.visible = true;
        const show = easeInOut(clamp((p - .50) / .30, 0, 1));
        this.palletOut.position.lerpVectors(this.palletOutOff, this.palletOutHome, show);
        this.moveBatteryNatural(
          this.rebuiltBattery,
          this.pos.beltOut,
          this.pos.palletOut,
          toPallet,
          time,
          'belt_to_pallet'
        );
      }

    } else if (phase === 'process_complete') {
      this.inspectPlatform.visible = false;
      this.palletOut.visible = true;
      this.palletOut.position.copy(this.palletOutHome);
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.palletOut);
      this.rebuiltBattery.rotation.set(0, 0, 0);
    }
  }

  setPhaseText(text) { $('phaseText').textContent = text; }
  setFade(opacity) { if (this.fadeEl) this.fadeEl.style.opacity = String(clamp(opacity, 0, 1)); }
  setCounts() {
    $('openedCount').textContent = String(this.openedCount);
    $('rebuiltCount').textContent = String(this.rebuiltCount);
    $('goodCellPool').textContent = `${this.goodPool} / 4`;
  }
  showMenu(show) {
    $('menuOverlay').classList.toggle('visible', show);
    $('menuOverlay').classList.toggle('hidden', !show);
  }
  showInspection(show) {
    $('inspectionOverlay').classList.toggle('visible', show);
    $('inspectionOverlay').classList.toggle('hidden', !show);
  }

  openSequence() {
    if (this.mode !== 'menu') return;
    this.mode = 'opening';
    this.transition = { name: 'open_lid_and_zoom', t: 0, duration: 1.65 };
    this.showMenu(false);
    this.showInspection(false);
    this.inspectWorld.visible = false;
    this.setFade(0);
    $('cellInfoCard').classList.add('hidden');
  }

  exitSequence() {
    if (this.mode !== 'inspection') return;
    this.mode = 'exiting';
    this.transition = { name: 'exit_fade_to_black', t: 0, duration: .85 };
    $('cellInfoCard').classList.add('hidden');
    this.setPhaseText('배터리 외부로 나가는 중');
  }

  generateCells() {
    const cells = [];
    for (let i = 0; i < 4; i++) {
      const r = Math.random();
      const appearance = r < 0.55 ? '정상' : (r < 0.80 ? '부푼 셀' : '터진 셀');
      let voltage;
      if (appearance === '정상') voltage = 7 + Math.random() * 5; // 7~12V, 10V 이상 합격 가능성 높임
      else if (appearance === '부푼 셀') voltage = Math.random() * 10.5;
      else voltage = Math.random() * 8.5;
      voltage = Number(voltage.toFixed(2));
      const good = appearance === '정상' && voltage >= 10.0;
      cells.push({ appearance, voltage, good });
    }
    return cells;
  }

  resetBatteryForNewCycle(full = false) {
    if (full) this.buildCaseCount = 0;
    this.inspectBattery.visible = true;
    this.inspectBattery.position.copy(this.pos.palletIn);
    this.inspectBattery.rotation.set(0, 0, 0);
    setBatteryOpen(this.inspectBattery, 0);
    this.rebuiltBattery.visible = false;
    this.rebuiltBattery.position.copy(this.pos.table);
    this.rebuiltBattery.rotation.set(0, 0, 0);
    setBatteryOpen(this.rebuiltBattery, 1);
    this.driver.visible = false;
    this.cellPedestal.visible = false;
    this.cellBackGlow.material.opacity = 0;
    this.inspectPlatform.material.opacity = 0.0;
    this.inspectPlatform.position.y = -0.75;
    this.palletIn.position.copy(this.palletInHome);
    this.palletOut.position.copy(this.palletOutOff);
    this.syncBuildCaseVisuals();
    this.removeCurrentCell();
  }

  beginCycle() {
    this.openedCount += 1;
    this.setCounts();
    this.resetBatteryForNewCycle();
    this.cycle = {
      phase: 'pallet_to_conveyor',
      timer: 0,
      cells: this.generateCells(),
      currentIndex: -1,
      batteryOrder: this.openedCount,
      createdNewBattery: false,
      resumePending: false,
      resumeIndex: null,
      goodBefore: this.goodPool,
    };
    this.setPhaseText('팔레트 → 컨베이어 이송 중');
    $('cellInfoCard').classList.add('hidden');
  }

  removeCurrentCell() {
    if (this.currentCellVisual) {
      this.inspectWorld.remove(this.currentCellVisual);
      this.currentCellVisual.traverse((o) => {
        if (o.geometry) o.geometry.dispose?.();
        if (o.material) {
          if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose?.());
          else o.material.dispose?.();
        }
      });
      this.currentCellVisual = null;
    }
  }

  startCell(index) {
    this.removeCurrentCell();
    const data = this.cycle.cells[index];
    this.currentCellVisual = createCell();
    this.currentCellVisual.position.set(0, 1.34, 2.2);
    this.currentCellVisual.scale.setScalar(0.0);
    this.currentCellVisual.rotation.y = -0.35;
    this.inspectWorld.add(this.currentCellVisual);
    this.cellShowProgress = 0;
    this.cycle.goodPlacement = null;

    $('batteryOrder').textContent = `배터리 #${this.cycle.batteryOrder}`;
    $('cellOrder').textContent = `CELL ${index + 1} / 4`;
    $('cellVoltage').textContent = `${data.voltage.toFixed(2)} V`;
    $('cellAppearance').textContent = data.appearance;
    $('cellJudgement').textContent = data.good ? '합격 셀 / 재사용 가능' : '불합격 셀 / 폐기 대상';
    $('voltageHint').textContent = data.voltage >= 10.0 ? '기준 전압 이상' : '기준 전압 미달';
    const appearanceBadge = $('appearanceBadge');
    const judgementBadge = $('judgementBadge');
    appearanceBadge.className = 'status-badge small ' + (data.appearance === '정상' ? 'good' : (data.appearance === '부푼 셀' ? 'warn' : 'bad'));
    appearanceBadge.textContent = data.appearance;
    judgementBadge.className = 'status-badge ' + (data.good ? 'good' : 'bad');
    judgementBadge.textContent = data.good ? '합격' : '불합격';
    $('cellInfoCard').classList.remove('hidden');

    if (data.good) {
      const slotIndex = this.buildCaseCount;
      this.goodPool += 1;
      this.buildCaseCount = Math.min(4, this.buildCaseCount + 1);
      this.rebuiltBattery.visible = false;
      this.cycle.goodPlacement = { slotIndex, target: this.getHiddenCaseTarget(slotIndex) };
      if (this.buildCaseCount >= 4) {
        this.rebuiltCount += 1;
        this.cycle.createdNewBattery = true;
        if (this.cycle.currentIndex < 3) {
          this.cycle.resumePending = true;
          this.cycle.resumeIndex = this.cycle.currentIndex + 1;
        }
      }
      this.setCounts();
    }
  }

  updateBelt(dt) {
    this.beltOffset += dt * 2.2;
    for (let i = 0; i < this.beltSlats.length; i++) {
      let x = ((-5.2 + i * 0.50 + this.beltOffset) % 11.0);
      x = x < 0 ? x + 11.0 : x;
      this.beltSlats[i].position.x = x - 5.5;
    }
  }

  placeDriverAtBattery(pack, screwIndex, descendT, spinT = 0) {
    const screws = pack.userData.boltPositions;
    const idx = clamp(screwIndex, 0, screws.length - 1);
    const local = screws[idx].clone();
    const world = local.applyMatrix4(pack.matrixWorld);
    this.driver.visible = true;
    const highY = world.y + 2.1;
    const lowY = world.y + 1.42;
    this.driver.position.set(world.x, lerp(highY, lowY, descendT), world.z);
    this.driver.rotation.y = spinT * Math.PI * 8;
  }

  animateDriverSequence(pack, elapsed, tightening = false) {
    const total = 4;
    const segment = 1.55;
    const moveDur = 0.28;
    const downDur = 0.34;
    const spinDur = 0.45;
    const upDur = 0.24;
    const totalDur = total * segment + 0.25;
    const seq = clamp(elapsed / totalDur, 0, 1);
    const screwIndex = Math.min(total - 1, Math.floor(elapsed / segment));
    const localT = clamp((elapsed - screwIndex * segment) / segment, 0, 1);
    const screws = pack.userData.boltPositions.map(v => v.clone().applyMatrix4(pack.matrixWorld));
    const curr = screws[screwIndex];
    const prev = screws[Math.max(0, screwIndex - 1)];
    let baseX = curr.x, baseZ = curr.z, y = curr.y + 2.1, spin = 0;

    if (screwIndex === 0 && localT < moveDur) {
      baseX = curr.x - 0.55 * (1 - localT / moveDur);
      baseZ = curr.z + 0.25 * (1 - localT / moveDur);
    } else if (localT < moveDur) {
      const p = easeInOut(localT / moveDur);
      baseX = lerp(prev.x, curr.x, p);
      baseZ = lerp(prev.z, curr.z, p);
      y = curr.y + 2.05 + Math.sin(p * Math.PI) * 0.12;
    } else if (localT < moveDur + downDur) {
      const p = easeInOut((localT - moveDur) / downDur);
      y = lerp(curr.y + 2.05, curr.y + 1.42, p);
    } else if (localT < moveDur + downDur + spinDur) {
      const p = (localT - moveDur - downDur) / spinDur;
      y = curr.y + 1.42;
      spin = (tightening ? -1 : 1) * p;
    } else {
      const p = easeInOut(clamp((localT - moveDur - downDur - spinDur) / upDur, 0, 1));
      y = lerp(curr.y + 1.42, curr.y + 2.05, p);
      spin = (tightening ? -1 : 1);
    }
    this.driver.visible = true;
    this.driver.position.set(baseX, y, baseZ);
    this.driver.rotation.y = spin * Math.PI * 8;
    return { totalDur, screwIndex, seq };
  }

  updateRebuildCellMotion(progress) {
    const anchors = [
      new THREE.Vector3(-0.9, 1.40, -0.35),
      new THREE.Vector3(-0.3, 1.42, 0.25),
      new THREE.Vector3(0.3, 1.42, -0.25),
      new THREE.Vector3(0.9, 1.40, 0.32),
    ];
    for (let i = 0; i < this.rebuildCells.length; i++) {
      const cell = this.rebuildCells[i];
      const start = i * 0.20;
      const local = clamp((progress - start) / 0.34, 0, 1);
      cell.visible = local > 0;
      if (!cell.visible) continue;
      const p = easeInOut(local);
      const from = anchors[i].clone().add(new THREE.Vector3(0, 1.2, 0));
      const to = anchors[i];
      cell.position.lerpVectors(from, to, p);
      cell.rotation.y = (1 - p) * 0.8;
      cell.scale.setScalar(0.34 + p * 0.24);
    }
  }

  getBuildSlotWorld(index) {
    const anchors = [
      new THREE.Vector3(-0.9, 1.40, -0.35),
      new THREE.Vector3(-0.3, 1.42, 0.25),
      new THREE.Vector3(0.3, 1.42, -0.25),
      new THREE.Vector3(0.9, 1.40, 0.32),
    ];
    return anchors[clamp(index, 0, anchors.length - 1)].clone();
  }

  getHiddenCaseTarget(index) {
    const targets = [
      new THREE.Vector3(3.05, 2.05, -1.05),
      new THREE.Vector3(3.25, 2.18, -0.55),
      new THREE.Vector3(3.45, 2.08, 0.10),
      new THREE.Vector3(3.65, 2.20, 0.65),
    ];
    return targets[clamp(index, 0, targets.length - 1)].clone();
  }

  syncBuildCaseVisuals() {
    for (let i = 0; i < this.rebuildCells.length; i++) {
      const cell = this.rebuildCells[i];
      cell.visible = false;
    }
  }

  moveBatteryNatural(obj, start, end, p, time, mode = 'belt') {
    const ep = easeInOut(clamp(p, 0, 1));
    obj.position.lerpVectors(start, end, ep);
    if (mode === 'pallet_to_belt') {
      obj.position.y = lerp(start.y, end.y, ep) + Math.sin(ep * Math.PI) * 0.10;
      obj.position.z += Math.sin(ep * Math.PI * 0.9) * 0.04;
      obj.rotation.z = -0.05 * Math.sin(ep * Math.PI);
      obj.rotation.x = 0.03 * Math.sin(ep * Math.PI);
    } else if (mode === 'belt') {
      obj.position.y = lerp(start.y, end.y, ep) + Math.sin(ep * Math.PI) * 0.035 + Math.sin(time * 7.4) * 0.010;
      obj.position.z += Math.sin(ep * Math.PI * 1.4) * 0.045 + Math.sin(time * 2.2) * 0.01;
      obj.rotation.z = 0.032 * Math.sin(ep * Math.PI * 1.6);
      obj.rotation.x = -0.018 * Math.sin(ep * Math.PI * 1.2);
    } else if (mode === 'belt_to_pallet') {
      obj.position.y = lerp(start.y, end.y, ep) + Math.sin(ep * Math.PI) * 0.08;
      obj.position.z += Math.sin(ep * Math.PI) * 0.05;
      obj.rotation.z = 0.05 * Math.sin(ep * Math.PI);
      obj.rotation.x = -0.025 * Math.sin(ep * Math.PI);
    }
  }

  updateInspection(dt, time) {
    if (!this.cycle) return;

    const phase = this.cycle.phase;
    const tablePhases = ['unscrew', 'opening_battery', 'cell_sequence', 'discard_motion', 'assemble_prepare', 'rebuild_cells', 'close_new_battery', 'screw_in_new', 'resume_case_prepare', 'resume_opening'];
    const isTablePhase = tablePhases.includes(phase);
    let tableReveal = isTablePhase ? 1 : 0;
    if (phase === 'conveyor_to_table') tableReveal = clamp((this.cycle.timer / 5.6 - 0.68) / 0.24, 0, 1);
    if (phase === 'new_to_conveyor') tableReveal = 1 - clamp((this.cycle.timer / 3.8) / 0.30, 0, 1);
    this.inspectPlatform.visible = tableReveal > 0.01;
    this.inspectPlatform.material.opacity = THREE.MathUtils.lerp(this.inspectPlatform.material.opacity, tableReveal, 0.08);
    this.inspectPlatform.position.y = THREE.MathUtils.lerp(this.inspectPlatform.position.y, -0.75 + 1.69 * tableReveal, 0.10);
    this.backgroundPanel.material.color.set(isTablePhase ? 0x132334 : 0x0c1620);
    this.labPanel.material.opacity = THREE.MathUtils.lerp(this.labPanel.material.opacity, isTablePhase || tableReveal > 0.01 ? 0.78 : 0.08, 0.08);

    if (phase === 'pallet_to_conveyor') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 3.2, 0, 1);
      this.setPhaseText('팔레트에서 컨베이어로 이동 중');
      this.moveBatteryNatural(this.inspectBattery, this.pos.palletIn, this.pos.beltStart, p, time, 'pallet_to_belt');
      this.palletIn.position.lerpVectors(this.palletInHome, this.palletInHome, 1);
      if (p >= 1) { this.cycle.phase = 'conveyor_to_table'; this.cycle.timer = 0; }

    } else if (phase === 'conveyor_to_table') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 5.6, 0, 1);
      this.setPhaseText('컨베이어 벨트 이동 중');
      this.updateBelt(dt);
      const hold = this.pos.table.clone();
      hold.y += 0.18;
      const travelP = clamp(p / 0.74, 0, 1);
      const settleP = clamp((p - 0.74) / 0.26, 0, 1);
      if (p < 0.74) {
        this.moveBatteryNatural(this.inspectBattery, this.pos.beltStart, hold, travelP, time, 'belt');
      } else {
        this.moveBatteryNatural(this.inspectBattery, hold, this.pos.table, settleP, time, 'belt');
        this.inspectBattery.rotation.x = THREE.MathUtils.lerp(this.inspectBattery.rotation.x, 0, 0.18 + settleP * 0.22);
        this.inspectBattery.rotation.y = THREE.MathUtils.lerp(this.inspectBattery.rotation.y, 0, 0.18 + settleP * 0.22);
        this.inspectBattery.rotation.z = THREE.MathUtils.lerp(this.inspectBattery.rotation.z, 0, 0.18 + settleP * 0.22);
      }
      const camP = easeInOut(p);
      this.camera.position.lerpVectors(new THREE.Vector3(-3.6, 3.0, 9.4), new THREE.Vector3(0.25, 3.05, 7.55), camP);
      this.camCurrentTarget.lerpVectors(new THREE.Vector3(-3.3, 1.2, 0), new THREE.Vector3(0.0, 1.35, 0), camP);
      this.camera.lookAt(this.camCurrentTarget);
      const palletHide = clamp((p - 0.10) / 0.36, 0, 1);
      this.palletIn.position.lerpVectors(this.palletInHome, this.palletInOff, easeInOut(palletHide));
      if (p >= 1) { this.inspectBattery.rotation.set(0, 0, 0); this.cycle.phase = 'unscrew'; this.cycle.timer = 0; }

    } else if (phase === 'unscrew') {
      this.cycle.timer += dt;
      const { totalDur, screwIndex, seq } = this.animateDriverSequence(this.inspectBattery, this.cycle.timer, false);
      this.inspectBattery.rotation.x = THREE.MathUtils.lerp(this.inspectBattery.rotation.x, 0, 0.22);
      this.inspectBattery.rotation.y = THREE.MathUtils.lerp(this.inspectBattery.rotation.y, 0, 0.22);
      this.inspectBattery.rotation.z = THREE.MathUtils.lerp(this.inspectBattery.rotation.z, 0, 0.22);
      this.setPhaseText(`드라이버 정밀 나사 해체 중 · ${Math.min(screwIndex + 1, 4)} / 4`);
      if (seq >= 1) { this.driver.visible = false; this.cycle.phase = 'opening_battery'; this.cycle.timer = 0; }

    } else if (phase === 'opening_battery') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 3.6, 0, 1);
      this.setPhaseText('뚜껑 분리 및 배터리 팩 오픈 중');
      setBatteryOpen(this.inspectBattery, easeInOut(p));
      this.insideGlow.intensity = 0.8 + p * 2.0;
      this.renderer.toneMappingExposure = 1.0 + p * 0.18;
      if (p >= 1) {
        this.cycle.phase = 'cell_sequence';
        this.cycle.timer = 0;
        this.cycle.currentIndex = 0;
        this.cellPedestal.visible = true;
        this.startCell(0);
      }

    } else if (phase === 'cell_sequence') {
      this.cycle.timer += dt;
      const cellData = this.cycle.cells[this.cycle.currentIndex];
      this.setPhaseText(`셀 분리 후 검사 진행 중 · ${this.cycle.currentIndex + 1} / 4`);
      this.cellShowProgress = clamp(this.cellShowProgress + dt * 1.2, 0, 1);
      const cp = easeInOut(this.cellShowProgress);
      if (this.currentCellVisual) {
        if (cellData.good && this.cycle.goodPlacement && this.cycle.timer > 1.35) {
          const flyP = easeInOut(clamp((this.cycle.timer - 1.35) / 1.10, 0, 1));
          const from = new THREE.Vector3(0, lerp(1.34, 1.82, cp), lerp(2.2, 2.92, cp));
          const target = this.cycle.goodPlacement.target;
          const to = target.clone();
          to.y += Math.sin(flyP * Math.PI) * 0.55;
          this.currentCellVisual.scale.setScalar(1.42 - flyP * 0.92);
          this.currentCellVisual.position.lerpVectors(from, to, flyP);
          this.currentCellVisual.rotation.y += dt * (3.4 - flyP * 2.2);
          this.cellBackGlow.material.opacity = 0.18 + flyP * 0.34 + Math.sin(time * 9.5) * 0.03;
          if (flyP >= 0.98 && !this.cycle.goodPlacement.done) {
            this.cycle.goodPlacement.done = true;
            this.syncBuildCaseVisuals();
          }
          this.setPhaseText(`새로운 케이스에 배치 중 · ${this.buildCaseCount} / 4`);
        } else {
          this.currentCellVisual.scale.setScalar(0.72 + cp * 0.72);
          this.currentCellVisual.position.set(0, lerp(1.34, 1.78, cp), lerp(2.2, 2.8, cp));
          this.currentCellVisual.rotation.y += dt * 0.55;
          this.cellBackGlow.material.opacity = 0.12 + Math.sin(time * 3.0) * 0.04;
        }
      }
      if (this.cycle.createdNewBattery && this.cycle.timer >= 2.1) {
        this.setPhaseText('양품 셀 4개 확보 완료 · 새 배터리 조립 준비');
      }
      if (this.cycle.timer >= 3.25) {
        this.cycle.timer = 0;
        this.removeCurrentCell();
        if (this.cycle.createdNewBattery) {
          this.cycle.phase = 'discard_motion';
          this.cellPedestal.visible = false;
          $('cellInfoCard').classList.add('hidden');
        } else {
          this.cycle.currentIndex += 1;
          if (this.cycle.currentIndex <= 3) {
            this.startCell(this.cycle.currentIndex);
          } else {
            this.cycle.phase = 'discard_motion';
            this.cellPedestal.visible = false;
            $('cellInfoCard').classList.add('hidden');
            this.cycle.timer = 0;
          }
        }
      }

    } else if (phase === 'discard_motion') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 3.8, 0, 1);
      this.setPhaseText('검사 완료 배터리 폐기 이동 중');
      const discardEnd = new THREE.Vector3(this.pos.discard.x + 1.4, this.pos.discard.y, this.pos.discard.z - 0.9);
      this.inspectBattery.position.lerpVectors(this.pos.table, discardEnd, easeInOut(p));
      this.inspectBattery.rotation.z = -p * 0.42;
      if (p >= 1) {
        this.inspectBattery.visible = false;
        if (this.cycle.createdNewBattery) {
          this.cycle.phase = 'close_new_battery';
        } else {
          this.beginCycle();
        }
        this.cycle.timer = 0;
      }

    } else if (phase === 'assemble_prepare') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 0.9, 0, 1);
      this.setPhaseText('새 배터리 케이스 준비 중');
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.table);
      setBatteryOpen(this.rebuiltBattery, 1);
      this.insideGlow.intensity = 1.2;
      if (p >= 1) { this.cycle.phase = 'rebuild_cells'; this.cycle.timer = 0; }

    } else if (phase === 'rebuild_cells') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 1.8, 0, 1);
      this.setPhaseText('양품 셀 4개를 새 배터리에 재배치 중');
      this.updateRebuildCellMotion(p);
      if (p >= 1) { this.cycle.phase = 'close_new_battery'; this.cycle.timer = 0; }

    } else if (phase === 'close_new_battery') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 2.1, 0, 1);
      this.setPhaseText('새 배터리 조립 완료 · 뚜껑 닫는 중');
      this.rebuiltBattery.visible = true;
      this.rebuiltBattery.position.copy(this.pos.table);
      setBatteryOpen(this.rebuiltBattery, 1 - easeInOut(p));
      if (p > 0.70) this.rebuildCells.forEach((c) => c.visible = false);
      if (p >= 1) { this.cycle.phase = 'screw_in_new'; this.cycle.timer = 0; }

    } else if (phase === 'screw_in_new') {
      this.cycle.timer += dt;
      const { totalDur, screwIndex, seq } = this.animateDriverSequence(this.rebuiltBattery, this.cycle.timer, true);
      this.setPhaseText(`드라이버 정밀 나사 체결 중 · ${Math.min(screwIndex + 1, 4)} / 4`);
      if (seq >= 1) { this.driver.visible = false; this.cycle.phase = 'new_to_conveyor'; this.cycle.timer = 0; }

    } else if (phase === 'new_to_conveyor') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 2.8, 0, 1);
      this.setPhaseText('새 배터리 컨베이어로 이송 중');
      this.updateBelt(dt);
      this.moveBatteryNatural(this.rebuiltBattery, this.pos.table, this.pos.beltOut, p, time, 'belt');
      if (p >= 1) { this.cycle.phase = 'conveyor_to_pallet'; this.cycle.timer = 0; }

    } else if (phase === 'conveyor_to_pallet') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 3.6, 0, 1);
      this.setPhaseText('새 배터리 팔레트 적재 중');
      this.updateBelt(dt);
      const palletShow = clamp((p - 0.10) / 0.40, 0, 1);
      this.palletOut.position.lerpVectors(this.palletOutOff, this.palletOutHome, easeInOut(palletShow));
      this.moveBatteryNatural(this.rebuiltBattery, this.pos.beltOut, this.pos.palletOut, p, time, 'belt_to_pallet');
      if (p >= 1) {
        this.cycle.phase = 'next_ready';
        this.cycle.timer = 0;
      }

    } else if (phase === 'next_ready') {
      this.cycle.timer += dt;
      this.setPhaseText(this.cycle.resumePending ? '새 배터리 적재 완료 · 남은 셀 검사 재개 준비' : '새 배터리 적재 완료 · 다음 검사 배터리 준비');
      if (this.cycle.timer >= 1.6) {
        this.buildCaseCount = 0;
        this.goodPool = 0;
        this.syncBuildCaseVisuals();
        if (this.cycle.resumePending && this.cycle.resumeIndex !== null) {
          this.rebuiltBattery.visible = false;
          this.inspectBattery.visible = true;
          this.inspectBattery.rotation.set(0, 0, 0);
          setBatteryOpen(this.inspectBattery, 0);
          this.inspectBattery.position.set(this.pos.beltStart.x - 1.2, this.pos.beltStart.y, this.pos.beltStart.z);
          this.cycle.phase = 'resume_case_prepare';
          this.cycle.timer = 0;
        } else {
          this.beginCycle();
        }
      }

    } else if (phase === 'resume_case_prepare') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 4.2, 0, 1);
      this.setPhaseText('새 검사 배터리 케이스 작업대 배치 중');
      const from = new THREE.Vector3(this.pos.beltStart.x - 1.6, this.pos.beltStart.y, this.pos.beltStart.z);
      const hold = this.pos.table.clone();
      hold.y += 0.18;
      const travelP = clamp(p / 0.76, 0, 1);
      const settleP = clamp((p - 0.76) / 0.24, 0, 1);
      if (p < 0.76) {
        this.moveBatteryNatural(this.inspectBattery, from, hold, travelP, time, 'belt');
      } else {
        this.moveBatteryNatural(this.inspectBattery, hold, this.pos.table, settleP, time, 'belt');
        this.inspectBattery.rotation.x = THREE.MathUtils.lerp(this.inspectBattery.rotation.x, 0, 0.18 + settleP * 0.22);
        this.inspectBattery.rotation.y = THREE.MathUtils.lerp(this.inspectBattery.rotation.y, 0, 0.18 + settleP * 0.22);
        this.inspectBattery.rotation.z = THREE.MathUtils.lerp(this.inspectBattery.rotation.z, 0, 0.18 + settleP * 0.22);
      }
      const camP = easeInOut(p);
      this.camera.position.lerpVectors(new THREE.Vector3(4.8, 3.0, 8.4), new THREE.Vector3(1.1, 3.0, 7.5), camP);
      this.camCurrentTarget.lerpVectors(new THREE.Vector3(3.0, 1.25, 0.0), new THREE.Vector3(0.0, 1.35, 0.0), camP);
      this.camera.lookAt(this.camCurrentTarget);
      if (p >= 1) {
        this.inspectBattery.rotation.set(0, 0, 0);
        this.cycle.phase = 'resume_opening';
        this.cycle.timer = 0;
      }

    } else if (phase === 'resume_opening') {
      this.cycle.timer += dt;
      const p = clamp(this.cycle.timer / 2.0, 0, 1);
      this.setPhaseText('남은 셀 검사를 위해 배터리 케이스 오픈 중');
      setBatteryOpen(this.inspectBattery, easeInOut(p));
      if (p >= 1) {
        this.cycle.createdNewBattery = false;
        this.cycle.resumePending = false;
        this.cycle.currentIndex = this.cycle.resumeIndex;
        this.cycle.resumeIndex = null;
        this.cycle.phase = 'cell_sequence';
        this.cycle.timer = 0;
        this.cellPedestal.visible = true;
        this.startCell(this.cycle.currentIndex);
      }
    }
  }

  updateTransition(dt) {
    if (!this.transition) return;
    this.transition.t = clamp(this.transition.t + dt / this.transition.duration, 0, 1);
    const p = easeInOut(this.transition.t);
    const name = this.transition.name;

    if (name === 'open_lid_and_zoom') {
      const camPos = new THREE.Vector3().copy(this.menuCam.pos).lerp(this.zoomCam.pos, p);
      const target = new THREE.Vector3().copy(this.menuCam.target).lerp(this.zoomCam.target, p);
      this.camera.position.copy(camPos);
      this.camCurrentTarget.copy(target);
      this.camera.lookAt(this.camCurrentTarget);
      setBatteryOpen(this.menuBattery, p);
      this.menuBattery.rotation.y = .22 + p * .10;
      this.menuBattery.position.y = .2 - p * .05;
      this.setFade(0);
      if (this.transition.t >= 1) this.transition = { name: 'open_fade_to_black', t: 0, duration: 1.15 };

    } else if (name === 'open_fade_to_black') {
      const camPos = new THREE.Vector3().copy(this.zoomCam.pos).lerp(new THREE.Vector3(.25, 1.55, 1.15), p);
      const target = new THREE.Vector3().copy(this.zoomCam.target).lerp(new THREE.Vector3(0, .7, 0), p);
      this.camera.position.copy(camPos);
      this.camCurrentTarget.copy(target);
      this.camera.lookAt(this.camCurrentTarget);
      this.setFade(p);
      if (this.transition.t >= 1) {
        this.menuBattery.visible = false;
        this.inspectWorld.visible = true;
        this.showInspection(true);
        this.showMenu(false);
        this.camera.position.copy(this.inspectCam.pos);
        this.camCurrentTarget.copy(this.inspectCam.target);
        this.camera.lookAt(this.camCurrentTarget);
        this.renderer.toneMappingExposure = 1.0;
        this.insideGlow.intensity = .65;
        this.prepareLiveView();
        this.transition = { name: 'open_fade_from_black', t: 0, duration: 1.20 };
      }

    } else if (name === 'open_fade_from_black') {
      this.setFade(1 - p);
      if (this.transition.t >= 1) {
        this.setFade(0);
        this.transition = null;
        this.mode = 'inspection';
      }

    } else if (name === 'exit_fade_to_black') {
      this.setFade(p);
      if (this.transition.t >= 1) {
        this.cycle = null;
        this.removeCurrentCell();
        this.cellPedestal.visible = false;
        this.cellBackGlow.material.opacity = 0;
        this.labPanel.material.opacity = 0;
        this.inspectWorld.visible = false;
        this.showInspection(false);
        this.resetBatteryForNewCycle(true);
        this.menuBattery.visible = true;
        setBatteryOpen(this.menuBattery, 0);
        this.menuBattery.rotation.y = .22;
        this.menuBattery.position.y = .2;
        this.camera.position.copy(this.menuCam.pos);
        this.camCurrentTarget.copy(this.menuCam.target);
        this.camera.lookAt(this.camCurrentTarget);
        this.renderer.toneMappingExposure = 1.0;
        this.showMenu(true);
        this.transition = { name: 'exit_fade_from_black', t: 0, duration: 1.20 };
      }

    } else if (name === 'exit_fade_from_black') {
      this.setFade(1 - p);
      if (this.transition.t >= 1) {
        this.setFade(0);
        this.transition = null;
        this.mode = 'menu';
        this.backendResetInProgress = false;
        this.setPhaseText('시스템 대기');
      }
    }
  }

  resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  animate() {
    this.renderer.setAnimationLoop(() => {
      const dt = Math.min(this.clock.getDelta(), 0.05);
      const time = this.clock.elapsedTime;
      this.updateTransition(dt);
      const now = performance.now();
      const holdUntil = Math.max(this.resultHoldUntil, this.motionHoldUntil);

      if (
        this.resultHoldUntil > 0 &&
        now >= this.resultHoldUntil &&
        this.liveStatus &&
        this.liveStatus.phase === 'cell_sequence' &&
        this.liveStatus.judgement
      ) {
        this.expiredResultSignature = this.lastStatusSignature;
        this.resultHoldUntil = 0;
        $('cellInfoCard').classList.add('hidden');
      }

      if (this.pendingStatus && now >= holdUntil) {
        const next = this.pendingStatus;
        this.pendingStatus = null;
        this.resultHoldUntil = 0;
        this.motionHoldUntil = 0;
        this.applyBackendStatus(next);
      }
      // V16는 임의 타이머/난수로 공정을 진행하지 않고 실제 백엔드 로그만 반영한다.
      if (this.mode === 'inspection') this.updateLiveVisuals(dt, time);
      if (this.mode === 'menu') {
        this.menuBattery.rotation.y = 0.22 + Math.sin(time * 0.45) * 0.04;
        this.menuBattery.position.y = 0.2 + Math.sin(time * 0.9) * 0.06;
        this.camera.position.copy(this.menuCam.pos);
        this.camCurrentTarget.copy(this.menuCam.target);
        this.camera.lookAt(this.camCurrentTarget);
      }
      this.renderer.render(this.scene, this.camera);
    });
  }
}

new StoryUI();
