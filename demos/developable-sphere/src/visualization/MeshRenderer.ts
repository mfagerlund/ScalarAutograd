import * as THREE from 'three';
import { TriangleMesh } from '../mesh/TriangleMesh';

export interface VertexClassification {
  hingeVertices: number[];
  seamVertices: number[];
}

export class MeshRenderer {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private meshObject: THREE.Mesh | null = null;
  private edgesObject: THREE.LineSegments | null = null;
  private controls: { isDragging: boolean; previousMousePosition: { x: number; y: number } };
  private rotation: { x: number; y: number };

  constructor(canvas: HTMLCanvasElement) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);

    this.camera = new THREE.PerspectiveCamera(
      60,
      canvas.width / canvas.height,
      0.1,
      1000
    );
    this.camera.position.z = 3;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setSize(canvas.width, canvas.height);
    this.renderer.setPixelRatio(window.devicePixelRatio);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(5, 10, 7.5);
    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-5, -5, -5);

    this.scene.add(ambientLight, directionalLight1, directionalLight2);

    // Mouse controls
    this.controls = {
      isDragging: false,
      previousMousePosition: { x: 0, y: 0 },
    };
    this.rotation = { x: 0, y: 0 };

    this.setupMouseControls(canvas);
  }

  private setupMouseControls(canvas: HTMLCanvasElement): void {
    canvas.addEventListener('mousedown', (e) => {
      this.controls.isDragging = true;
      this.controls.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e) => {
      if (this.controls.isDragging) {
        const deltaX = e.clientX - this.controls.previousMousePosition.x;
        const deltaY = e.clientY - this.controls.previousMousePosition.y;

        this.rotation.y += deltaX * 0.005;
        this.rotation.x += deltaY * 0.005;

        if (this.meshObject) {
          this.meshObject.rotation.y = this.rotation.y;
          this.meshObject.rotation.x = this.rotation.x;
        }
        if (this.edgesObject) {
          this.edgesObject.rotation.y = this.rotation.y;
          this.edgesObject.rotation.x = this.rotation.x;
        }

        this.controls.previousMousePosition = { x: e.clientX, y: e.clientY };
        this.render();
      }
    });

    canvas.addEventListener('mouseup', () => {
      this.controls.isDragging = false;
    });

    canvas.addEventListener('mouseleave', () => {
      this.controls.isDragging = false;
    });

    // Zoom with mouse wheel
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.position.z += e.deltaY * 0.001;
      this.camera.position.z = Math.max(1.5, Math.min(10, this.camera.position.z));
      this.render();
    });
  }

  updateMesh(mesh: TriangleMesh, classification: VertexClassification): void {
    // Remove old mesh
    if (this.meshObject) {
      this.scene.remove(this.meshObject);
      this.meshObject.geometry.dispose();
      if (Array.isArray(this.meshObject.material)) {
        this.meshObject.material.forEach((m) => m.dispose());
      } else {
        this.meshObject.material.dispose();
      }
    }

    // Remove old edges
    if (this.edgesObject) {
      this.scene.remove(this.edgesObject);
      this.edgesObject.geometry.dispose();
      if (Array.isArray(this.edgesObject.material)) {
        this.edgesObject.material.forEach((m) => m.dispose());
      } else {
        (this.edgesObject.material as THREE.Material).dispose();
      }
    }

    // Create THREE.js geometry
    const geometry = new THREE.BufferGeometry();

    // Positions
    const positions = new Float32Array(mesh.vertices.length * 3);
    mesh.vertices.forEach((v, i) => {
      positions[3 * i] = v.x.data;
      positions[3 * i + 1] = v.y.data;
      positions[3 * i + 2] = v.z.data;
    });
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Indices
    const indices: number[] = [];
    mesh.faces.forEach((f) => {
      indices.push(f.vertices[0], f.vertices[1], f.vertices[2]);
    });
    geometry.setIndex(indices);

    // Vertex colors (blue = hinge, red/orange = seam)
    const colors = new Float32Array(mesh.vertices.length * 3);
    const seamSet = new Set(classification.seamVertices);
    for (let i = 0; i < mesh.vertices.length; i++) {
      const isSeam = seamSet.has(i);
      colors[3 * i] = isSeam ? 1.0 : 0.3; // R
      colors[3 * i + 1] = isSeam ? 0.4 : 0.6; // G
      colors[3 * i + 2] = isSeam ? 0.2 : 1.0; // B
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    // Compute normals
    geometry.computeVertexNormals();

    // Material
    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
      shininess: 30,
    });

    this.meshObject = new THREE.Mesh(geometry, material);
    this.meshObject.rotation.x = this.rotation.x;
    this.meshObject.rotation.y = this.rotation.y;
    this.scene.add(this.meshObject);

    // Add hinge edges (crease lines)
    this.addHingeEdges(mesh, classification);
  }

  private addHingeEdges(mesh: TriangleMesh, classification: VertexClassification): void {
    const seamSet = new Set(classification.seamVertices);
    const edgeSet = new Set<string>();
    const edgePositions: number[] = [];

    // Find all edges on the boundary between hinge and seam regions (crease lines)
    mesh.faces.forEach((face) => {
      const v0 = face.vertices[0];
      const v1 = face.vertices[1];
      const v2 = face.vertices[2];

      // Check each edge
      const edges = [
        [v0, v1],
        [v1, v2],
        [v2, v0],
      ];

      edges.forEach(([a, b]) => {
        // Highlight edges where vertices have different classifications (boundary/crease)
        const aIsSeam = seamSet.has(a);
        const bIsSeam = seamSet.has(b);

        if (aIsSeam !== bIsSeam) {
          // Create edge key (sorted to avoid duplicates)
          const key = a < b ? `${a}-${b}` : `${b}-${a}`;
          if (!edgeSet.has(key)) {
            edgeSet.add(key);

            const vA = mesh.vertices[a];
            const vB = mesh.vertices[b];

            edgePositions.push(
              vA.x.data, vA.y.data, vA.z.data,
              vB.x.data, vB.y.data, vB.z.data
            );
          }
        }
      });
    });

    // Create edge geometry
    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(edgePositions), 3)
    );

    // Bright cyan/yellow color for hinge edges
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: 0xffff00, // Bright yellow
      linewidth: 2,
      transparent: true,
      opacity: 0.8,
    });

    this.edgesObject = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    this.edgesObject.rotation.x = this.rotation.x;
    this.edgesObject.rotation.y = this.rotation.y;
    this.scene.add(this.edgesObject);
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.render();
  }
}
