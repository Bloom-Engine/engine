/**
 * Bloom Scene Graph — Retained-mode 3D scene management.
 *
 * Unlike immediate-mode drawing (drawCube, drawModel), the scene graph holds
 * persistent meshes that survive across frames. Systems update geometry and
 * transforms; the renderer draws all visible nodes each frame automatically.
 *
 * Designed for architecture/CAD editors compiled natively via Perry.
 */

// ============================================================
// FFI declarations (match package.json nativeLibrary.functions)
// ============================================================

declare function bloom_scene_create_node(): number;
declare function bloom_scene_destroy_node(handle: number): void;
declare function bloom_scene_set_visible(handle: number, visible: number): void;
declare function bloom_scene_set_cast_shadow(handle: number, cast: number): void;
declare function bloom_scene_set_receive_shadow(handle: number, receive: number): void;
declare function bloom_scene_set_parent(handle: number, parent: number): void;
declare function bloom_scene_set_transform(handle: number, matrix: number): void;
declare function bloom_scene_update_geometry(
  handle: number,
  vertices: number,
  vertexCount: number,
  indices: number,
  indexCount: number,
): void;
declare function bloom_scene_set_material_color(handle: number, r: number, g: number, b: number, a: number): void;
declare function bloom_scene_set_material_pbr(handle: number, roughness: number, metalness: number): void;
declare function bloom_scene_set_material_texture(handle: number, textureIdx: number): void;
declare function bloom_scene_node_count(): number;

// Frame callbacks
declare function bloom_register_frame_callback(priority: number, callback: number): number;
declare function bloom_unregister_frame_callback(id: number): void;

// Multiple lights
declare function bloom_add_directional_light(
  dx: number, dy: number, dz: number,
  r: number, g: number, b: number,
  intensity: number,
): void;
declare function bloom_add_point_light(
  x: number, y: number, z: number, range: number,
  r: number, g: number, b: number,
  intensity: number,
): void;

// Shadows
declare function bloom_enable_shadows(): void;
declare function bloom_disable_shadows(): void;

// Model attachment
declare function bloom_scene_attach_model(nodeHandle: number, modelHandle: number, meshIndex: number): void;

// Scene picking
declare function bloom_scene_pick(screenX: number, screenY: number): number;
declare function bloom_pick_hit_handle(): number;
declare function bloom_pick_hit_distance(): number;
declare function bloom_pick_hit_x(): number;
declare function bloom_pick_hit_y(): number;
declare function bloom_pick_hit_z(): number;
declare function bloom_pick_hit_normal_x(): number;
declare function bloom_pick_hit_normal_y(): number;
declare function bloom_pick_hit_normal_z(): number;

// Geometry generation
declare function bloom_scene_extrude_polygon(
  handle: number,
  polygon: number,
  pointCount: number,
  depth: number,
): void;
declare function bloom_scene_subtract_box(
  handle: number,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): void;

// ============================================================
// Types
// ============================================================

export type SceneNodeHandle = number;

export interface PbrMaterial {
  color?: [number, number, number];
  roughness?: number;
  metalness?: number;
  opacity?: number;
  textureIdx?: number;
}

// ============================================================
// Public API
// ============================================================

/**
 * Create a new empty scene node. Returns a handle.
 * The node is visible by default but has no geometry.
 */
export function createSceneNode(): SceneNodeHandle {
  return bloom_scene_create_node();
}

/**
 * Destroy a scene node, freeing its GPU resources.
 */
export function destroySceneNode(handle: SceneNodeHandle): void {
  bloom_scene_destroy_node(handle);
}

/**
 * Set visibility of a scene node.
 */
export function setSceneNodeVisible(handle: SceneNodeHandle, visible: boolean): void {
  bloom_scene_set_visible(handle, visible ? 1 : 0);
}

/**
 * Set whether this node casts shadows.
 */
export function setSceneNodeCastShadow(handle: SceneNodeHandle, cast: boolean): void {
  bloom_scene_set_cast_shadow(handle, cast ? 1 : 0);
}

/**
 * Set whether this node receives shadows.
 */
export function setSceneNodeReceiveShadow(handle: SceneNodeHandle, receive: boolean): void {
  bloom_scene_set_receive_shadow(handle, receive ? 1 : 0);
}

/**
 * Set the parent of a scene node. Pass 0 for no parent (root node).
 */
export function setSceneNodeParent(handle: SceneNodeHandle, parent: SceneNodeHandle): void {
  bloom_scene_set_parent(handle, parent);
}

/**
 * Set the 4x4 transform matrix for a scene node.
 * Matrix is in column-major order (same as Three.js/glm).
 */
export function setSceneNodeTransform(handle: SceneNodeHandle, matrix: number[]): void {
  bloom_scene_set_transform(handle, matrix as any);
}

/**
 * Update the geometry of a scene node.
 *
 * @param vertices — Flat array of vertex data. Each vertex has 12 floats:
 *   [x, y, z, nx, ny, nz, r, g, b, a, u, v]
 * @param indices — Flat array of triangle indices (3 per triangle).
 */
export function updateSceneNodeGeometry(
  handle: SceneNodeHandle,
  vertices: number[],
  indices: number[],
): void {
  bloom_scene_update_geometry(
    handle,
    vertices as any,
    vertices.length / 12,
    indices as any,
    indices.length,
  );
}

/**
 * Set the material color and opacity of a scene node.
 * Color components are 0-1 (not 0-255).
 */
export function setSceneNodeColor(handle: SceneNodeHandle, r: number, g: number, b: number, a: number = 1): void {
  bloom_scene_set_material_color(handle, r, g, b, a);
}

/**
 * Set PBR material properties (roughness and metalness).
 */
export function setSceneNodePbr(handle: SceneNodeHandle, roughness: number, metalness: number): void {
  bloom_scene_set_material_pbr(handle, roughness, metalness);
}

/**
 * Set the texture for a scene node's material.
 * Pass 0 for the default white texture.
 */
export function setSceneNodeTexture(handle: SceneNodeHandle, textureIdx: number): void {
  bloom_scene_set_material_texture(handle, textureIdx);
}

/**
 * Get the number of live scene nodes.
 */
export function getSceneNodeCount(): number {
  return bloom_scene_node_count();
}

// ============================================================
// Frame Callbacks (replaces R3F's useFrame)
// ============================================================

/**
 * Register a callback that runs every frame, ordered by priority.
 * Lower priority numbers run first (matching R3F convention:
 * SlabSystem=1, ItemSystem=2, DoorSystem=3, WallSystem=4, RoofSystem=5).
 *
 * Returns an ID that can be used to unregister the callback.
 */
export function registerFrameCallback(priority: number, callback: (deltaTime: number) => void): number {
  return bloom_register_frame_callback(priority, callback as any);
}

/**
 * Unregister a frame callback by ID.
 */
export function unregisterFrameCallback(id: number): void {
  bloom_unregister_frame_callback(id);
}

// ============================================================
// Multiple Lights
// ============================================================

/**
 * Add a directional light. Color is 0-1 range. Up to 4 additional directional lights.
 * Called each frame (lights are cleared at frame start).
 */
export function addDirectionalLight(
  dx: number, dy: number, dz: number,
  r: number, g: number, b: number,
  intensity: number,
): void {
  bloom_add_directional_light(dx, dy, dz, r, g, b, intensity);
}

/**
 * Add a point light. Color is 0-1 range. Up to 16 point lights.
 * Called each frame (lights are cleared at frame start).
 */
// ============================================================
// Shadow Mapping
// ============================================================

/**
 * Enable shadow mapping (single directional light).
 * Shadows are rendered from the primary directional light's perspective.
 */
export function enableShadows(): void {
  bloom_enable_shadows();
}

/**
 * Disable shadow mapping.
 */
export function disableShadows(): void {
  bloom_disable_shadows();
}

// ============================================================
// Model Attachment (GLTF → Scene Node)
// ============================================================

/**
 * Attach a loaded GLTF model's mesh to a scene node.
 * Copies vertex/index data from the model into the scene node's geometry.
 * This is the native equivalent of R3F's useGLTF + Clone.
 *
 * @param nodeHandle — scene node to receive the geometry
 * @param modelHandle — loaded model handle (from loadModel())
 * @param meshIndex — which mesh in the model (0 for single-mesh models)
 */
export function attachModelToNode(
  nodeHandle: SceneNodeHandle,
  modelHandle: number,
  meshIndex: number = 0,
): void {
  bloom_scene_attach_model(nodeHandle, modelHandle, meshIndex);
}

// ============================================================
// Scene Picking (raycasting from screen)
// ============================================================

export interface PickHit {
  hit: boolean;
  handle: SceneNodeHandle;
  distance: number;
  point: { x: number; y: number; z: number };
  normal: { x: number; y: number; z: number };
}

/**
 * Raycast from screen coordinates against all visible scene nodes.
 * Returns the closest hit with world-space point and normal.
 *
 * Must be called after beginMode3D() so the camera matrices are set.
 */
export function pickScene(screenX: number, screenY: number): PickHit {
  const hit = bloom_scene_pick(screenX, screenY) !== 0;
  if (!hit) {
    return {
      hit: false,
      handle: 0,
      distance: 0,
      point: { x: 0, y: 0, z: 0 },
      normal: { x: 0, y: 0, z: 0 },
    };
  }
  return {
    hit: true,
    handle: bloom_pick_hit_handle(),
    distance: bloom_pick_hit_distance(),
    point: {
      x: bloom_pick_hit_x(),
      y: bloom_pick_hit_y(),
      z: bloom_pick_hit_z(),
    },
    normal: {
      x: bloom_pick_hit_normal_x(),
      y: bloom_pick_hit_normal_y(),
      z: bloom_pick_hit_normal_z(),
    },
  };
}

// ============================================================
// Geometry Generation
// ============================================================

/**
 * Extrude a 2D polygon along Y axis and set as the node's geometry.
 * The polygon is a flat array of 2D points: [x0, z0, x1, z1, ...].
 * Extrusion goes from Y=0 to Y=depth.
 */
export function extrudePolygon(
  handle: SceneNodeHandle,
  polygon: number[],
  depth: number,
): void {
  bloom_scene_extrude_polygon(handle, polygon as any, polygon.length / 2, depth);
}

/**
 * Subtract an axis-aligned box from a scene node's geometry.
 * Removes triangles fully inside the box (simplified CSG for door cutouts).
 */
export function subtractBox(
  handle: SceneNodeHandle,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): void {
  bloom_scene_subtract_box(handle, minX, minY, minZ, maxX, maxY, maxZ);
}

export function addPointLight(
  x: number, y: number, z: number, range: number,
  r: number, g: number, b: number,
  intensity: number,
): void {
  bloom_add_point_light(x, y, z, range, r, g, b, intensity);
}
