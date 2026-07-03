/**
 * Bloom Scene Graph — Retained-mode 3D scene management.
 *
 * Unlike immediate-mode drawing (drawCube, drawModel), the scene graph holds
 * persistent meshes that survive across frames. Systems update geometry and
 * transforms; the renderer draws all visible nodes each frame automatically.
 *
 * Use immediate-mode for prototypes and small scenes; use the scene graph
 * when you need persistence, picking, per-node materials, or hundreds of
 * objects (nodes are frustum- and occlusion-culled automatically).
 *
 * Coordinate system: right-handed, Y-up, world units are meters.
 * Surface colors are 0-255 per channel; light colors are 0-1 floats with
 * a separate intensity (see addDirectionalLight).
 *
 * Designed for architecture/CAD editors compiled natively via Perry.
 */

// ============================================================
// FFI declarations (match package.json nativeLibrary.functions)
// ============================================================

declare function bloom_scene_create_node(): number;
declare function bloom_scene_destroy_node(handle: number): void;
declare function bloom_scene_set_visible(handle: number, visible: number): void;
declare function bloom_scene_set_gi_only(handle: number, gi_only: number): void;
declare function bloom_scene_set_cast_shadow(handle: number, cast: number): void;
declare function bloom_scene_set_receive_shadow(handle: number, receive: number): void;
declare function bloom_scene_set_parent(handle: number, parent: number): void;
declare function bloom_scene_set_transform(handle: number, matrix: number): void;
declare function bloom_scene_set_trs(handle: number, px: number, py: number, pz: number, yaw: number, scale: number): void;
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
declare function bloom_dump_shadow_map(path: number): void;

// Post-processing
declare function bloom_enable_postfx(): void;
declare function bloom_disable_postfx(): void;
declare function bloom_postfx_set_selected(handle: number): void;
declare function bloom_postfx_set_hovered(handle: number): void;
declare function bloom_postfx_set_outline_color(r: number, g: number, b: number, a: number): void;
declare function bloom_postfx_set_outline_thickness(thickness: number): void;

// Model attachment
declare function bloom_scene_attach_model(nodeHandle: number, modelHandle: number, meshIndex: number): void;

// 3D→2D Projection
declare function bloom_project_to_screen(wx: number, wy: number, wz: number): number;
declare function bloom_project_screen_y(): number;

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
// Q6: multi-hit picking
declare function bloom_scene_pick_all(screenX: number, screenY: number, maxResults: number): number;
declare function bloom_pick_all_handle(index: number): number;
declare function bloom_pick_all_distance(index: number): number;

// Q8: water material
declare function bloom_scene_set_material_water(handle: number, waveAmp: number, waveSpeed: number, r: number, g: number, b: number, a: number): void;

// Q4: transform read-back
declare function bloom_scene_get_transform(handle: number, index: number): number;
// Q5: node bounds
declare function bloom_scene_get_bounds_min_x(handle: number): number;
declare function bloom_scene_get_bounds_min_y(handle: number): number;
declare function bloom_scene_get_bounds_min_z(handle: number): number;
declare function bloom_scene_get_bounds_max_x(handle: number): number;
declare function bloom_scene_get_bounds_max_y(handle: number): number;
declare function bloom_scene_get_bounds_max_z(handle: number): number;
// Q7: user data
declare function bloom_scene_set_user_data(handle: number, data: number): void;
declare function bloom_scene_get_user_data(handle: number): number;

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
 * Mark a node as a GI proxy: it feeds the global-illumination inputs
 * (ray-tracing BLAS/TLAS, mesh cards, SDF clipmap) but is skipped by the
 * main render, planar reflections, and the sun-shadow pass.
 *
 * Use this when your world renders through the material system (custom
 * WGSL materials, no scene nodes): register invisible duplicates of the
 * big static geometry — terrain, buildings, tree trunks — so SSGI picks
 * up bounce light from surfaces that are off-screen. Set an approximate
 * flat base colour via setSceneNodeMaterialPbr so the bounce carries the
 * right hue; leave the node visible (the flag handles exclusion).
 */
export function setSceneNodeGiOnly(handle: SceneNodeHandle, giOnly: boolean): void {
  bloom_scene_set_gi_only(handle, giOnly ? 1 : 0);
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
 *
 * NOTE: this crosses the FFI as an array-into-i64-pointer, which Perry
 * 0.5.x rejects at runtime ("Expected safe integer for native i64
 * parameter"). Until the scratch-buffer migration lands, prefer
 * `setSceneNodeTrs` for position/yaw/scale placement — it is all-scalar
 * and works on every Perry.
 */
export function setSceneNodeTransform(handle: SceneNodeHandle, matrix: number[]): void {
  bloom_scene_set_transform(handle, matrix as any);
}

/**
 * Place a scene node with position + Y-axis rotation + uniform scale.
 * All-scalar FFI (Perry 0.5.x safe). Yaw is radians, same convention as
 * `drawModelRotated`.
 */
export function setSceneNodeTrs(
  handle: SceneNodeHandle,
  x: number, y: number, z: number,
  yaw: number = 0,
  scale: number = 1,
): void {
  bloom_scene_set_trs(handle, x, y, z, yaw, scale);
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
/**
 * Set a node's material base color.
 *
 * Color components are 0-255 (the engine-wide convention — same scale as
 * the `Colors` presets and every draw* call). Before v0.5 this function
 * alone took 0-1 floats; passing a `Colors` constant silently rendered
 * white. Values are clamped; alpha defaults to opaque.
 */
export function setSceneNodeColor(handle: SceneNodeHandle, r: number, g: number, b: number, a: number = 255): void {
  bloom_scene_set_material_color(handle, r / 255, g / 255, b / 255, a / 255);
}

/**
 * Set PBR material properties (roughness and metalness).
 */
/**
 * Add a reduced-detail geometry variant to a node. The node's normal
 * geometry is the finest level; `lodIndex` 0,1,2… are progressively
 * coarser variants. `maxCoverage` is the screen-coverage threshold
 * (0..1, fraction of the viewport the node's bounds span) below which
 * this variant renders — give coarser variants smaller values, e.g.
 * LOD0 at 0.4, LOD1 at 0.15, LOD2 at 0.05. Selection has hysteresis to
 * avoid flicker at boundaries. Shadows and picking always use the full
 * geometry. Vertex layout matches updateSceneNodeGeometry (12 floats).
 */
export function setSceneNodeLod(
  handle: SceneNodeHandle, lodIndex: number,
  vertices: number[], indices: number[], maxCoverage: number,
): void {
  bloom_scene_set_lod(handle, lodIndex, vertices as any, vertices.length / 12, indices as any, indices.length, maxCoverage);
}

/**
 * Attach a mesh from a loaded model as a reduced-detail variant — the
 * model-based counterpart of setSceneNodeLod. Export your LOD meshes as
 * separate meshes (or models) and attach each with its threshold.
 */
export function attachModelLodToNode(
  node: SceneNodeHandle, model: { handle: number }, meshIndex: number,
  lodIndex: number, maxCoverage: number,
): void {
  bloom_scene_attach_model_lod(node, model.handle, meshIndex, lodIndex, maxCoverage);
}

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
/**
 * Add a directional light. Light color is a 0-1 float per channel with a
 * separate intensity multiplier (radiometric convention, like Unity and
 * Unreal — light colors are not surface colors and may meaningfully
 * exceed 1.0 via intensity). Direction need not be normalized.
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

/**
 * Dump the shadow map depth texture to a grayscale PNG for debugging.
 */
export function dumpShadowMap(path: string): void {
  bloom_dump_shadow_map(path as any);
}

// ============================================================
// Post-Processing (Outlines, SSAO)
// ============================================================

/**
 * Enable the post-processing pipeline (outlines, SSAO).
 * Must be called after initWindow.
 */
export function enablePostFx(): void {
  bloom_enable_postfx();
}

/**
 * Disable post-processing (render directly to screen).
 */
export function disablePostFx(): void {
  bloom_disable_postfx();
}

/**
 * Set the selected scene node for outline rendering.
 * Pass 0 to clear selection. Matching Pascal Editor's outliner.selectedObjects.
 */
export function setPostFxSelected(handle: SceneNodeHandle): void {
  bloom_postfx_set_selected(handle);
}

/**
 * Set the hovered scene node for outline rendering.
 * Pass 0 to clear hover.
 */
export function setPostFxHovered(handle: SceneNodeHandle): void {
  bloom_postfx_set_hovered(handle);
}

/**
 * Set the outline color for selected objects (0-1 range).
 */
/** Selection-outline color, 0-255 per channel (engine-wide convention). */
export function setOutlineColor(r: number, g: number, b: number, a: number = 255): void {
  bloom_postfx_set_outline_color(r / 255, g / 255, b / 255, a / 255);
}

/**
 * Set the outline thickness in pixels.
 */
export function setOutlineThickness(thickness: number): void {
  bloom_postfx_set_outline_thickness(thickness);
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
// 3D → 2D Projection (for UI overlays in 3D space)
// ============================================================

/**
 * Project a world-space 3D point to screen coordinates.
 * Returns { x, y, visible }. If the point is behind the camera, visible is false.
 *
 * This is the native equivalent of R3F's drei Html component positioning.
 * Used for zone labels, dimension references, interactive controls.
 */
export function projectToScreen(
  worldX: number, worldY: number, worldZ: number,
): { x: number; y: number; visible: boolean } {
  const sx = bloom_project_to_screen(worldX, worldY, worldZ);
  const sy = bloom_project_screen_y();
  return {
    x: sx,
    y: sy,
    visible: sx > -9000,
  };
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

// ============================================================
// Q4: Transform Read-back
// ============================================================

/**
 * Read back the 4x4 transform matrix of a scene node as a flat 16-element
 * array in column-major order (same layout as setSceneNodeTransform).
 */
export function getSceneNodeTransform(handle: SceneNodeHandle): number[] {
  const m: number[] = new Array(16);
  for (let i = 0; i < 16; i++) {
    m[i] = bloom_scene_get_transform(handle, i);
  }
  return m;
}

// ============================================================
// Q5: Scene Node Bounds
// ============================================================

/**
 * Return the cached AABB of a scene node's geometry in local space.
 * Recomputed automatically when geometry is updated via updateSceneNodeGeometry.
 */
export function getSceneNodeBounds(handle: SceneNodeHandle): { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } } {
  return {
    min: {
      x: bloom_scene_get_bounds_min_x(handle),
      y: bloom_scene_get_bounds_min_y(handle),
      z: bloom_scene_get_bounds_min_z(handle),
    },
    max: {
      x: bloom_scene_get_bounds_max_x(handle),
      y: bloom_scene_get_bounds_max_y(handle),
      z: bloom_scene_get_bounds_max_z(handle),
    },
  };
}

// ============================================================
// Q7: Scene Node User Data
// ============================================================

/**
 * Attach an arbitrary integer to a scene node. Used by editors to associate
 * entity ids with scene nodes so picking can return the entity id directly.
 */
export function setSceneNodeUserData(handle: SceneNodeHandle, data: number): void {
  bloom_scene_set_user_data(handle, data);
}

export function getSceneNodeUserData(handle: SceneNodeHandle): number {
  return bloom_scene_get_user_data(handle);
}

// ============================================================
// Q6: Multi-Hit Picking
// ============================================================

/**
 * Raycast against all visible scene nodes and return all hits sorted by
 * distance (closest first). Used by editors for Alt-click cycling through
 * occluded objects. Returns an array of { handle, distance }.
 */
export function pickSceneAll(screenX: number, screenY: number, maxResults: number = 8): { handle: number; distance: number }[] {
  const count = bloom_scene_pick_all(screenX, screenY, maxResults);
  const results: { handle: number; distance: number }[] = [];
  for (let i = 0; i < count; i++) {
    results.push({
      handle: bloom_pick_all_handle(i),
      distance: bloom_pick_all_distance(i),
    });
  }
  return results;
}

// ============================================================
// Q8: Water Material
// ============================================================

/**
 * Set a water-like material on a scene node with translucent tint and
 * low roughness. The wave parameters are stored for a future animated
 * water shader; currently they're visual-only placeholders.
 */
/**
 * Water surface material. Color is 0-255 per channel (engine-wide
 * convention; was 0-1 before v0.5).
 */
export function setSceneNodeWaterMaterial(
  handle: SceneNodeHandle,
  waveAmplitude: number, waveSpeed: number,
  r: number, g: number, b: number, a: number,
): void {
  bloom_scene_set_material_water(handle, waveAmplitude, waveSpeed, r / 255, g / 255, b / 255, a / 255);
}

export function addPointLight(
  x: number, y: number, z: number, range: number,
  r: number, g: number, b: number,
  intensity: number,
): void {
  bloom_add_point_light(x, y, z, range, r, g, b, intensity);
}
