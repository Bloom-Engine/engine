// BloomWatchApp — SwiftUI @main for bloom-backed watchOS games.
//
// Compiled by Perry when --features watchos-swift-app is on. Owns the process
// entry (@main), spawns the game thread calling _perry_user_main, and renders
// the bloom draw list through a SwiftUI Canvas. Input (Digital Crown + taps)
// feeds back into bloom via the C hooks in lib.rs.

import SwiftUI
import Foundation
import SceneKit
import ImageIO

// MARK: - FFI into the Rust side

// Perry's --features watchos-swift-app renames C-level `main` → `perry_user_main`
// (one Mach-O underscore is added by the linker; Swift's @_silgen_name adds
// another, matching the `__perry_user_main` actually emitted in the object).
@_silgen_name("_perry_user_main") func _perry_user_main()

// Inbound (Swift → Rust) — input and layout
@_silgen_name("bloom_watchos_crown_delta") func bloom_watchos_crown_delta(_ delta: Double)
@_silgen_name("bloom_watchos_touch") func bloom_watchos_touch(_ idx: Int64, _ x: Double, _ y: Double, _ active: Int64)
@_silgen_name("bloom_watchos_set_screen") func bloom_watchos_set_screen(_ w: Double, _ h: Double)
@_silgen_name("bloom_watchos_set_bundle_path") func bloom_watchos_set_bundle_path(_ path: UnsafePointer<CChar>)

// Outbound (Rust → Swift) — draw list snapshot
@_silgen_name("bloom_watchos_frame_count") func bloom_watchos_frame_count() -> UInt64
@_silgen_name("bloom_watchos_copy_draw_list") func bloom_watchos_copy_draw_list(_ dst: UnsafeMutablePointer<DrawCmd>, _ max: Int64) -> Int64
@_silgen_name("bloom_watchos_clear_color") func bloom_watchos_clear_color(_ out: UnsafeMutablePointer<Double>)
@_silgen_name("bloom_watchos_texture_path") func bloom_watchos_texture_path(_ handle: UInt32) -> UnsafePointer<CChar>?

// 3D camera + gate
@_silgen_name("bloom_watchos_camera_state") func bloom_watchos_camera_state(_ out: UnsafeMutablePointer<Double>)
@_silgen_name("bloom_watchos_has_3d") func bloom_watchos_has_3d() -> Double

// Retained scene graph accessors
@_silgen_name("bloom_watchos_scene_copy_nodes") func bloom_watchos_scene_copy_nodes(_ dst: UnsafeMutablePointer<SceneNodeInfo>, _ max: Int64) -> Int64
@_silgen_name("bloom_watchos_scene_copy_lights") func bloom_watchos_scene_copy_lights(_ dst: UnsafeMutablePointer<SceneLight>, _ max: Int64) -> Int64
@_silgen_name("bloom_watchos_scene_geometry") func bloom_watchos_scene_geometry(_ handle: UInt32, _ out: UnsafeMutablePointer<SceneGeometryPtrs>)
@_silgen_name("bloom_watchos_scene_skin") func bloom_watchos_scene_skin(_ handle: UInt32, _ out: UnsafeMutablePointer<SceneSkinPtrs>)

struct SceneSkinPtrs {
    var jointHandles: UnsafePointer<UInt32>? = nil
    var jointCount: UInt32 = 0
    var inverseBind: UnsafePointer<Float>? = nil
    var inverseBindMatrixCount: UInt32 = 0
    var vertexJoints: UnsafePointer<UInt32>? = nil
    var vertexJointCount: UInt32 = 0
    var vertexWeights: UnsafePointer<Float>? = nil
    var vertexWeightCount: UInt32 = 0
}

// Post-FX state
@_silgen_name("bloom_watchos_postfx_state") func bloom_watchos_postfx_state(_ out: UnsafeMutablePointer<PostFxState>)

struct PostFxState {
    var enabled: UInt32 = 1
    var autoExposure: UInt32 = 0
    var vignetteStrength: Float = 0
    var vignetteSoftness: Float = 0
    var chromaticAberration: Float = 0
    var filmGrain: Float = 0
    var exposure: Float = 1
    var _pad: Float = 0
}

// SceneNodeInfo — must match Rust's #[repr(C)] struct in scene.rs.
struct SceneNodeInfo {
    var handle: UInt32 = 0
    var parent: UInt32 = 0
    var visible: UInt32 = 0
    var geometryVersion: UInt32 = 0

    var color: (Float, Float, Float, Float) = (1, 1, 1, 1)
    var roughness: Float = 0
    var metalness: Float = 0
    var texture: UInt32 = 0
    var hasGeometry: UInt32 = 0

    var transform: (
        Float, Float, Float, Float,
        Float, Float, Float, Float,
        Float, Float, Float, Float,
        Float, Float, Float, Float
    ) = (1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1)

    // PBR texture slots (0 = unset).
    var texBaseColor: UInt32 = 0
    var texNormal: UInt32 = 0
    var texMetallicRoughness: UInt32 = 0
    var texEmissive: UInt32 = 0
    var texOcclusion: UInt32 = 0
    // Bumps when bloom sets skin data on this node. 0 = not skinned.
    var skinVersion: UInt32 = 0
}

struct SceneLight {
    var kind: UInt32 = 0
    var _pad: UInt32 = 0
    var posOrDir: (Float, Float, Float) = (0, 0, 0)
    var range: Float = 0
    var color: (Float, Float, Float) = (1, 1, 1)
    var intensity: Float = 0
}

struct SceneGeometryPtrs {
    var positions: UnsafePointer<Float>? = nil
    var positionCount: UInt32 = 0
    var normals: UnsafePointer<Float>? = nil
    var normalCount: UInt32 = 0
    var uvs: UnsafePointer<Float>? = nil
    var uvCount: UInt32 = 0
    var indices: UnsafePointer<UInt32>? = nil
    var indexCount: UInt32 = 0
}

// MARK: - DrawCmd mirror (must match draw_list.rs #[repr(C)])

let TEXT_CAP = 256

struct DrawCmd {
    var kind: Int32 = 0
    var _pad0: Int32 = 0
    var tex: UInt32 = 0
    var _pad1: UInt32 = 0
    var x: Double = 0, y: Double = 0, w: Double = 0, h: Double = 0
    var srcX: Double = 0, srcY: Double = 0, srcW: Double = 0, srcH: Double = 0
    var ox: Double = 0, oy: Double = 0
    var r: Double = 255, g: Double = 255, b: Double = 255, a: Double = 255
    var rot: Double = 0, size: Double = 0, thickness: Double = 1, _pad2: Double = 0
    // Inline UTF-8 text — 256 bytes expressed as a 32-tuple of UInt64.
    var t00: UInt64 = 0, t01: UInt64 = 0, t02: UInt64 = 0, t03: UInt64 = 0
    var t04: UInt64 = 0, t05: UInt64 = 0, t06: UInt64 = 0, t07: UInt64 = 0
    var t08: UInt64 = 0, t09: UInt64 = 0, t10: UInt64 = 0, t11: UInt64 = 0
    var t12: UInt64 = 0, t13: UInt64 = 0, t14: UInt64 = 0, t15: UInt64 = 0
    var t16: UInt64 = 0, t17: UInt64 = 0, t18: UInt64 = 0, t19: UInt64 = 0
    var t20: UInt64 = 0, t21: UInt64 = 0, t22: UInt64 = 0, t23: UInt64 = 0
    var t24: UInt64 = 0, t25: UInt64 = 0, t26: UInt64 = 0, t27: UInt64 = 0
    var t28: UInt64 = 0, t29: UInt64 = 0, t30: UInt64 = 0, t31: UInt64 = 0
    var textLen: UInt64 = 0
}

// Command kinds — keep in sync with draw_list::kind.
let K_RECT: Int32 = 1
let K_RECT_LINES: Int32 = 2
let K_CIRCLE: Int32 = 3
let K_CIRCLE_LINES: Int32 = 4
let K_LINE: Int32 = 5
let K_TRIANGLE: Int32 = 6
let K_TEXTURE: Int32 = 7
let K_TEXTURE_REC: Int32 = 8
let K_TEXTURE_PRO: Int32 = 9
let K_TEXT: Int32 = 10

// 3D primitives
let K_CUBE: Int32 = 20
let K_CUBE_WIRES: Int32 = 21
let K_SPHERE: Int32 = 22
let K_SPHERE_WIRES: Int32 = 23
let K_CYLINDER: Int32 = 24
let K_PLANE: Int32 = 25
let K_GRID: Int32 = 26

/// Decode the inline UTF-8 bytes at the struct's text region into a String.
/// Takes a pointer into the underlying draw-list buffer so we read the real
/// bytes, not a stack-copied value (Swift's withUnsafePointer(to:) on a let
/// field gives a local copy, which truncates anything past 8 bytes).
func cmdTextString(_ ptr: UnsafePointer<DrawCmd>) -> String {
    let n = Int(min(ptr.pointee.textLen, UInt64(TEXT_CAP)))
    if n == 0 { return "" }
    return UnsafeRawPointer(ptr)
        .advanced(by: MemoryLayout<DrawCmd>.offset(of: \.t00)!)
        .withMemoryRebound(to: UInt8.self, capacity: n) { p in
            let buf = UnsafeBufferPointer(start: p, count: n)
            return String(decoding: buf, as: UTF8.self)
        }
}

// MARK: - Texture cache (handle → CGImage)

final class TextureCache {
    static let shared = TextureCache()
    private var cache: [UInt32: CGImage] = [:]
    private let lock = NSLock()

    func image(for handle: UInt32) -> CGImage? {
        lock.lock()
        defer { lock.unlock() }
        if let img = cache[handle] { return img }
        guard let cpath = bloom_watchos_texture_path(handle) else { return nil }
        let path = String(cString: cpath)
        let url = URL(fileURLWithPath: path)
        // Use CGImageSource so we transparently load PNG, JPEG, and anything
        // else ImageIO understands — the glTF loader embeds JPEGs too.
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        cache[handle] = img
        return img
    }
}

// MARK: - @main

@main
struct BloomWatchApp: App {
    init() {
        // Hand bloom the bundle resource path so bloom_load_texture can
        // resolve relative asset paths. Must happen before the game thread
        // starts loading anything.
        if let res = Bundle.main.resourcePath {
            res.withCString { bloom_watchos_set_bundle_path($0) }
        }
        // Spawn the game thread. _perry_user_main blocks in runGame's while
        // loop, so this must not run on the main thread.
        let t = Thread(block: { _perry_user_main() })
        t.name = "bloom-game"
        t.stackSize = 2 * 1024 * 1024
        t.start()
    }

    var body: some Scene {
        WindowGroup {
            BloomRootView()
        }
    }
}

// MARK: - Root view

struct BloomRootView: View {
    // Accumulated crown value (Digital Crown is delta-based in SwiftUI).
    @State private var crown: Double = 0.0
    @State private var lastCrown: Double = 0.0
    // Frame tick — incremented when bloom's frame counter changes. Triggers
    // Canvas redraw.
    @State private var frameTick: UInt64 = 0
    // Reusable snapshot buffer to avoid per-frame allocation. 4096 cmds is
    // generous for a watch-sized game.
    @State private var drawBuf: UnsafeMutableBufferPointer<DrawCmd> = {
        let p = UnsafeMutablePointer<DrawCmd>.allocate(capacity: 4096)
        p.initialize(repeating: DrawCmd(), count: 4096)
        return UnsafeMutableBufferPointer(start: p, count: 4096)
    }()
    @State private var fx: PostFxState = PostFxState()

    let refresh = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // 3D layer — SceneView is always present so the first
                // bloom_begin_mode_3d call can light it up without view
                // reconstruction. The scene is empty (transparent) until a
                // 3D command arrives.
                BloomSceneView(frameTick: frameTick, drawBuf: drawBuf)
                    .ignoresSafeArea()

                // 2D overlay — Canvas drives drawing from the snapshot and
                // owns input because Canvas is the topmost responder.
                Canvas { ctx, size in
                    _ = frameTick

                    // Clear (only when there's no 3D layer — otherwise the
                    // SceneView's clear owns the background and this would
                    // paint over it).
                    if bloom_watchos_has_3d() < 0.5 {
                        var cc = [Double](repeating: 0, count: 4)
                        cc.withUnsafeMutableBufferPointer { buf in
                            bloom_watchos_clear_color(buf.baseAddress!)
                        }
                        ctx.fill(
                            Path(CGRect(origin: .zero, size: size)),
                            with: .color(Color(.sRGB,
                                red: cc[0] / 255.0,
                                green: cc[1] / 255.0,
                                blue: cc[2] / 255.0,
                                opacity: cc[3] / 255.0))
                        )
                    }

                    // Pull this frame's commands. 2D draws render here; 3D
                    // commands are filtered by BloomSceneView.
                    let n = Int(bloom_watchos_copy_draw_list(drawBuf.baseAddress!, 4096))
                    for i in 0..<n {
                        let ptr = drawBuf.baseAddress!.advanced(by: i)
                        let k = ptr.pointee.kind
                        if k >= 20 && k <= 29 { continue }  // 3D — handled by SceneView
                        drawOne(ctx: ctx, cmdPtr: ptr)
                    }
                }
            }
            .onAppear {
                bloom_watchos_set_screen(Double(geo.size.width), Double(geo.size.height))
            }
            .onChange(of: geo.size) { _, new in
                bloom_watchos_set_screen(Double(new.width), Double(new.height))
            }
            .focusable()
            .digitalCrownRotation(
                $crown,
                from: -1_000_000.0, through: 1_000_000.0, by: 0.001,
                sensitivity: .medium,
                isContinuous: true,
                isHapticFeedbackEnabled: false
            )
            .onChange(of: crown) { _, new in
                bloom_watchos_crown_delta(new - lastCrown)
                lastCrown = new
            }
            .onTapGesture(coordinateSpace: .local) { loc in
                bloom_watchos_touch(0, Double(loc.x), Double(loc.y), 1)
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                    bloom_watchos_touch(0, Double(loc.x), Double(loc.y), 0)
                }
            }
            .onReceive(refresh) { _ in
                let f = bloom_watchos_frame_count()
                if f != frameTick {
                    frameTick = f
                    withUnsafeMutablePointer(to: &fx) { bloom_watchos_postfx_state($0) }
                }
            }
            // Post-FX modifiers — applied only when fx.enabled is set. Each
            // effect also checks its own strength before costing anything.
            .brightness(fx.enabled != 0 && fx.autoExposure == 0 ? Double(fx.exposure - 1.0) : 0)
            .overlay {
                if fx.enabled != 0 && fx.vignetteStrength > 0.001 {
                    // Vignette via a radial gradient: transparent in the
                    // middle, black at the edges. Strength scales the outer
                    // alpha; softness shifts the gradient's transition point.
                    let soft = max(0.0, min(0.95, Double(fx.vignetteSoftness)))
                    let str = min(1.0, Double(fx.vignetteStrength))
                    Canvas { ctx, size in
                        let c = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
                        let r = max(size.width, size.height) * 0.75
                        let g = Gradient(stops: [
                            .init(color: .black.opacity(0), location: soft),
                            .init(color: .black.opacity(str), location: 1.0),
                        ])
                        ctx.fill(
                            Path(CGRect(origin: .zero, size: size)),
                            with: .radialGradient(g, center: c, startRadius: 0, endRadius: r)
                        )
                    }
                    .allowsHitTesting(false)
                }
            }
        }
        .ignoresSafeArea()
        .background(Color.black)
    }
}

// MARK: - 3D layer (SceneKit)

/// SceneView wrapper that rebuilds the SCNScene each frame from bloom's
/// immediate-mode 3D draw commands. Retained-mode scene graph (bloom_scene_*)
/// would use a delta-sync path — deferred until the retained API lands for
/// watchOS.
struct BloomSceneView: View {
    let frameTick: UInt64
    let drawBuf: UnsafeMutableBufferPointer<DrawCmd>

    @State private var scene: SCNScene = makeBaseScene()
    @State private var contentRoot: SCNNode = SCNNode()   // immediate-mode per-frame draws
    @State private var retainedRoot: SCNNode = SCNNode()  // bloom_scene_* retained nodes
    @State private var lightsRoot: SCNNode = SCNNode()    // bloom_add_*_light instances
    @State private var cameraNode: SCNNode = SCNNode()

    /// Cache of SCNNodes for retained scene graph, keyed by bloom handle.
    @State private var retainedNodes: [UInt32: SCNNode] = [:]
    /// Geometry cache keyed by (handle, geometryVersion) so unchanged meshes
    /// aren't rebuilt each frame.
    @State private var retainedGeomVersions: [UInt32: UInt32] = [:]
    /// Same pattern for skinning — we only rebuild SCNSkinner when the
    /// bloom-side skin_version changes, since SCNSkinner carries GPU
    /// resources and rebuilding every frame would churn them.
    @State private var retainedSkinVersions: [UInt32: UInt32] = [:]

    @State private var nodeBuf: UnsafeMutableBufferPointer<SceneNodeInfo> = {
        let p = UnsafeMutablePointer<SceneNodeInfo>.allocate(capacity: 1024)
        p.initialize(repeating: SceneNodeInfo(), count: 1024)
        return UnsafeMutableBufferPointer(start: p, count: 1024)
    }()
    @State private var lightBuf: UnsafeMutableBufferPointer<SceneLight> = {
        let p = UnsafeMutablePointer<SceneLight>.allocate(capacity: 64)
        p.initialize(repeating: SceneLight(), count: 64)
        return UnsafeMutableBufferPointer(start: p, count: 64)
    }()

    var body: some View {
        SceneView(
            scene: scene,
            pointOfView: cameraNode,
            options: [.rendersContinuously],
            preferredFramesPerSecond: 30,
            antialiasingMode: .none
        )
        .onAppear {
            if contentRoot.parent == nil {
                scene.rootNode.addChildNode(contentRoot)
            }
            if retainedRoot.parent == nil {
                scene.rootNode.addChildNode(retainedRoot)
            }
            if lightsRoot.parent == nil {
                scene.rootNode.addChildNode(lightsRoot)
            }
            if cameraNode.parent == nil {
                cameraNode.camera = SCNCamera()
                scene.rootNode.addChildNode(cameraNode)
            }
            rebuild()
        }
        .onChange(of: frameTick) { _, _ in rebuild() }
    }

    private func rebuild() {
        // If the game hasn't opened a 3D section, hide the scene entirely.
        if bloom_watchos_has_3d() < 0.5 {
            contentRoot.isHidden = true
            return
        }
        contentRoot.isHidden = false

        // Clear previous frame's primitives. Reusing the same contentRoot
        // avoids churn at the SCNScene.rootNode level.
        contentRoot.childNodes.forEach { $0.removeFromParentNode() }

        // Camera.
        var camS = [Double](repeating: 0, count: 11)
        camS.withUnsafeMutableBufferPointer { buf in bloom_watchos_camera_state(buf.baseAddress!) }
        let px = camS[0], py = camS[1], pz = camS[2]
        let tx = camS[3], ty = camS[4], tz = camS[5]
        let fovy = camS[9], proj = camS[10]
        cameraNode.position = SCNVector3(x: Float(px), y: Float(py), z: Float(pz))
        // Guard against look(at: self.position) which produces a NaN orientation.
        if abs(px - tx) < 1e-6 && abs(py - ty) < 1e-6 && abs(pz - tz) < 1e-6 {
            // No explicit target — point down -Z like a default game camera.
            cameraNode.eulerAngles = SCNVector3(0, 0, 0)
        } else {
            cameraNode.look(at: SCNVector3(x: Float(tx), y: Float(ty), z: Float(tz)))
        }
        if let cam = cameraNode.camera {
            cam.fieldOfView = CGFloat(fovy)
            cam.usesOrthographicProjection = proj < 0.5
            cam.zNear = 0.01
            cam.zFar = 10000
        }

        // Clear color for SceneKit background.
        var cc = [Double](repeating: 0, count: 4)
        cc.withUnsafeMutableBufferPointer { buf in bloom_watchos_clear_color(buf.baseAddress!) }
        scene.background.contents = UIColor(
            red: CGFloat(cc[0] / 255.0),
            green: CGFloat(cc[1] / 255.0),
            blue: CGFloat(cc[2] / 255.0),
            alpha: CGFloat(cc[3] / 255.0)
        )

        // Walk the draw list for 3D commands only.
        let n = Int(bloom_watchos_copy_draw_list(drawBuf.baseAddress!, 4096))
        for i in 0..<n {
            let cmd = drawBuf[i]
            if cmd.kind < 20 || cmd.kind > 29 { continue }
            if let node = makeSceneNode(for: cmd) {
                contentRoot.addChildNode(node)
            }
        }

        // Retained scene graph — delta-sync against the cached handle→SCNNode map.
        syncRetainedNodes()
        syncLights()
    }

    private func syncRetainedNodes() {
        let count = Int(bloom_watchos_scene_copy_nodes(nodeBuf.baseAddress!, 1024))
        var seen: Set<UInt32> = []
        seen.reserveCapacity(count)

        // Pass 1: ensure a SCNNode exists for every live handle and update its
        // transform + material. Geometry is versioned so we only rebuild when
        // it actually changes.
        for i in 0..<count {
            let info = nodeBuf[i]
            seen.insert(info.handle)

            let node: SCNNode
            if let existing = retainedNodes[info.handle] {
                node = existing
            } else {
                node = SCNNode()
                retainedNodes[info.handle] = node
                retainedRoot.addChildNode(node)
            }

            node.isHidden = info.visible == 0
            node.transform = scnMatrix4(from: info.transform)

            let cached = retainedGeomVersions[info.handle] ?? .max
            if info.hasGeometry != 0 && cached != info.geometryVersion {
                node.geometry = buildGeometry(handle: info.handle)
                retainedGeomVersions[info.handle] = info.geometryVersion
            }

            // Skinning: rebuild SCNSkinner when skin_version changes. Needs
            // all bone SCNNodes to exist first — the retained pass-1 loop
            // creates them before we reach this point for any later-visited
            // skinned mesh. For skinned meshes near the top of the scene
            // walk order, bones may still be missing on frame 0; we retry
            // the build on the next sync tick if any bone is still nil.
            if info.skinVersion != 0 && retainedSkinVersions[info.handle] != info.skinVersion {
                if let skinner = buildSkinner(mesh: node, handle: info.handle) {
                    node.skinner = skinner
                    retainedSkinVersions[info.handle] = info.skinVersion
                }
            }

            if let g = node.geometry {
                let m = g.firstMaterial ?? SCNMaterial()
                // PBR needs per-vertex normals. Some glTF exports (e.g. Fox)
                // omit NORMAL entirely — fall back to constant lighting so
                // the mesh still shows its base color / texture instead of
                // rendering pitch black.
                let hasNormals = g.sources(for: .normal).count > 0
                m.lightingModel = hasNormals ? .physicallyBased : .constant

                // Base color: texture wins over factor when present.
                let baseColorTex = info.texBaseColor != 0 ? info.texBaseColor : info.texture
                if baseColorTex != 0, let img = TextureCache.shared.image(for: baseColorTex) {
                    m.diffuse.contents = img
                } else {
                    m.diffuse.contents = UIColor(
                        red: CGFloat(info.color.0),
                        green: CGFloat(info.color.1),
                        blue: CGFloat(info.color.2),
                        alpha: CGFloat(info.color.3)
                    )
                }

                // Normal map.
                if info.texNormal != 0, let img = TextureCache.shared.image(for: info.texNormal) {
                    m.normal.contents = img
                } else {
                    m.normal.contents = nil
                }

                // Metallic-roughness: glTF packs metalness in B, roughness in G
                // into one texture. SceneKit's .metalness + .roughness channels
                // pointing at the same image each sample the correct channel
                // by convention when .textureComponents is set.
                if info.texMetallicRoughness != 0,
                   let img = TextureCache.shared.image(for: info.texMetallicRoughness) {
                    m.metalness.contents = img
                    m.metalness.textureComponents = .blue
                    m.roughness.contents = img
                    m.roughness.textureComponents = .green
                } else {
                    m.metalness.contents = NSNumber(value: info.metalness)
                    m.roughness.contents = NSNumber(value: info.roughness)
                }

                // Emissive.
                if info.texEmissive != 0, let img = TextureCache.shared.image(for: info.texEmissive) {
                    m.emission.contents = img
                } else {
                    m.emission.contents = nil
                }

                // AO.
                if info.texOcclusion != 0, let img = TextureCache.shared.image(for: info.texOcclusion) {
                    m.ambientOcclusion.contents = img
                } else {
                    m.ambientOcclusion.contents = nil
                }

                g.materials = [m]
            }
        }

        // Pass 2: retire SCNNodes whose handle no longer appears in the
        // snapshot (bloom_scene_destroy_node).
        for (h, node) in retainedNodes where !seen.contains(h) {
            node.removeFromParentNode()
            retainedNodes.removeValue(forKey: h)
            retainedGeomVersions.removeValue(forKey: h)
        }

        // Pass 3: fix up parent-child — done after all nodes exist so
        // bloom_scene_set_parent to a not-yet-created node doesn't matter.
        for i in 0..<count {
            let info = nodeBuf[i]
            guard let node = retainedNodes[info.handle] else { continue }
            let desiredParent: SCNNode = info.parent == 0 ? retainedRoot
                : (retainedNodes[info.parent] ?? retainedRoot)
            if node.parent !== desiredParent {
                node.removeFromParentNode()
                desiredParent.addChildNode(node)
            }
        }
    }

    private func buildGeometry(handle: UInt32) -> SCNGeometry? {
        var ptrs = SceneGeometryPtrs()
        withUnsafeMutablePointer(to: &ptrs) { bloom_watchos_scene_geometry(handle, $0) }
        guard let pos = ptrs.positions, let idx = ptrs.indices,
              ptrs.positionCount > 0, ptrs.indexCount > 0 else { return nil }

        let vcount = Int(ptrs.positionCount) / 3
        let posData = Data(bytes: pos, count: Int(ptrs.positionCount) * MemoryLayout<Float>.size)
        let posSrc = SCNGeometrySource(
            data: posData,
            semantic: .vertex,
            vectorCount: vcount,
            usesFloatComponents: true,
            componentsPerVector: 3,
            bytesPerComponent: MemoryLayout<Float>.size,
            dataOffset: 0,
            dataStride: 3 * MemoryLayout<Float>.size
        )

        var sources: [SCNGeometrySource] = [posSrc]
        if let nrm = ptrs.normals, ptrs.normalCount >= ptrs.positionCount {
            let nrmData = Data(bytes: nrm, count: Int(ptrs.normalCount) * MemoryLayout<Float>.size)
            sources.append(SCNGeometrySource(
                data: nrmData,
                semantic: .normal,
                vectorCount: vcount,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: 3 * MemoryLayout<Float>.size
            ))
        }
        if let uv = ptrs.uvs, ptrs.uvCount >= UInt32(vcount * 2) {
            let uvData = Data(bytes: uv, count: Int(ptrs.uvCount) * MemoryLayout<Float>.size)
            sources.append(SCNGeometrySource(
                data: uvData,
                semantic: .texcoord,
                vectorCount: vcount,
                usesFloatComponents: true,
                componentsPerVector: 2,
                bytesPerComponent: MemoryLayout<Float>.size,
                dataOffset: 0,
                dataStride: 2 * MemoryLayout<Float>.size
            ))
        }

        let idxData = Data(bytes: idx, count: Int(ptrs.indexCount) * MemoryLayout<UInt32>.size)
        let element = SCNGeometryElement(
            data: idxData,
            primitiveType: .triangles,
            primitiveCount: Int(ptrs.indexCount) / 3,
            bytesPerIndex: MemoryLayout<UInt32>.size
        )

        return SCNGeometry(sources: sources, elements: [element])
    }

    private func buildSkinner(mesh: SCNNode, handle: UInt32) -> SCNSkinner? {
        var ptrs = SceneSkinPtrs()
        withUnsafeMutablePointer(to: &ptrs) { bloom_watchos_scene_skin(handle, $0) }
        guard let jointsPtr = ptrs.jointHandles,
              let ibmPtr = ptrs.inverseBind,
              let vjPtr = ptrs.vertexJoints,
              let vwPtr = ptrs.vertexWeights,
              ptrs.jointCount > 0, ptrs.vertexJointCount > 0,
              let geom = mesh.geometry
        else { return nil }

        // Resolve bone SCNNodes from bloom handles. Skip if any is missing
        // — skinner without all bones would crash on draw.
        var bones: [SCNNode] = []
        bones.reserveCapacity(Int(ptrs.jointCount))
        for i in 0..<Int(ptrs.jointCount) {
            let h = jointsPtr[i]
            guard let node = retainedNodes[h] else { return nil }
            bones.append(node)
        }

        // Inverse-bind matrices wrapped as NSValue(SCNMatrix4).
        var ibmValues: [NSValue] = []
        ibmValues.reserveCapacity(Int(ptrs.inverseBindMatrixCount))
        for i in 0..<Int(ptrs.inverseBindMatrixCount) {
            let base = i * 16
            let m = SCNMatrix4(
                m11: ibmPtr[base+0],  m12: ibmPtr[base+1],  m13: ibmPtr[base+2],  m14: ibmPtr[base+3],
                m21: ibmPtr[base+4],  m22: ibmPtr[base+5],  m23: ibmPtr[base+6],  m24: ibmPtr[base+7],
                m31: ibmPtr[base+8],  m32: ibmPtr[base+9],  m33: ibmPtr[base+10], m34: ibmPtr[base+11],
                m41: ibmPtr[base+12], m42: ibmPtr[base+13], m43: ibmPtr[base+14], m44: ibmPtr[base+15]
            )
            ibmValues.append(NSValue(scnMatrix4: m))
        }

        // Bone weights + indices as SCNGeometrySources. 4 per vertex.
        let vertexCount = Int(ptrs.vertexWeightCount) / 4
        let weightsData = Data(bytes: vwPtr, count: Int(ptrs.vertexWeightCount) * MemoryLayout<Float>.size)
        let weightsSrc = SCNGeometrySource(
            data: weightsData, semantic: .boneWeights, vectorCount: vertexCount,
            usesFloatComponents: true, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<Float>.size, dataOffset: 0,
            dataStride: 4 * MemoryLayout<Float>.size
        )
        let indicesData = Data(bytes: vjPtr, count: Int(ptrs.vertexJointCount) * MemoryLayout<UInt32>.size)
        let indicesSrc = SCNGeometrySource(
            data: indicesData, semantic: .boneIndices, vectorCount: vertexCount,
            usesFloatComponents: false, componentsPerVector: 4,
            bytesPerComponent: MemoryLayout<UInt32>.size, dataOffset: 0,
            dataStride: 4 * MemoryLayout<UInt32>.size
        )

        return SCNSkinner(
            baseGeometry: geom,
            bones: bones,
            boneInverseBindTransforms: ibmValues,
            boneWeights: weightsSrc,
            boneIndices: indicesSrc
        )
    }

    private func syncLights() {
        let n = Int(bloom_watchos_scene_copy_lights(lightBuf.baseAddress!, 64))
        // Simpler than delta sync for lights — there are few of them and
        // bloom's intended use is setting them once on scene init. Rebuild
        // each frame.
        lightsRoot.childNodes.forEach { $0.removeFromParentNode() }
        for i in 0..<n {
            let l = lightBuf[i]
            let node = SCNNode()
            let light = SCNLight()
            light.color = UIColor(
                red: CGFloat(l.color.0),
                green: CGFloat(l.color.1),
                blue: CGFloat(l.color.2),
                alpha: 1.0
            )
            light.intensity = CGFloat(l.intensity * 1000.0)  // bloom's 0..1 → SceneKit lumens
            if l.kind == 1 {
                light.type = .directional
                // Point light forward along `pos_or_dir`.
                node.look(at: SCNVector3(
                    x: node.position.x + l.posOrDir.0,
                    y: node.position.y + l.posOrDir.1,
                    z: node.position.z + l.posOrDir.2
                ))
            } else if l.kind == 2 {
                light.type = .omni
                light.attenuationEndDistance = CGFloat(l.range)
                node.position = SCNVector3(x: l.posOrDir.0, y: l.posOrDir.1, z: l.posOrDir.2)
            }
            node.light = light
            lightsRoot.addChildNode(node)
        }
    }
}

/// Convert a column-major flat 16-float transform (bloom's layout) into an
/// SCNMatrix4. SCNMatrix4 is column-major too, so it's a direct mapping.
private func scnMatrix4(from m: (
    Float, Float, Float, Float,
    Float, Float, Float, Float,
    Float, Float, Float, Float,
    Float, Float, Float, Float
)) -> SCNMatrix4 {
    return SCNMatrix4(
        m11: m.0,  m12: m.1,  m13: m.2,  m14: m.3,
        m21: m.4,  m22: m.5,  m23: m.6,  m24: m.7,
        m31: m.8,  m32: m.9,  m33: m.10, m34: m.11,
        m41: m.12, m42: m.13, m43: m.14, m44: m.15
    )
}

private func makeBaseScene() -> SCNScene {
    let s = SCNScene()
    // Minimal default lighting so objects aren't pitch black before the game
    // adds its own lights.
    let ambient = SCNNode()
    ambient.light = SCNLight()
    ambient.light?.type = .ambient
    ambient.light?.intensity = 300
    s.rootNode.addChildNode(ambient)

    let dir = SCNNode()
    dir.light = SCNLight()
    dir.light?.type = .directional
    dir.light?.intensity = 800
    dir.eulerAngles = SCNVector3(-Float.pi / 3, Float.pi / 6, 0)
    s.rootNode.addChildNode(dir)
    return s
}

private func makeSceneNode(for cmd: DrawCmd) -> SCNNode? {
    let pos = SCNVector3(Float(cmd.x), Float(cmd.y), Float(cmd.srcX))
    let color = UIColor(
        red: CGFloat(cmd.r / 255.0),
        green: CGFloat(cmd.g / 255.0),
        blue: CGFloat(cmd.b / 255.0),
        alpha: CGFloat(cmd.a / 255.0)
    )

    let geo: SCNGeometry?
    switch cmd.kind {
    case K_CUBE, K_CUBE_WIRES:
        geo = SCNBox(width: CGFloat(cmd.w), height: CGFloat(cmd.h),
                     length: CGFloat(cmd.size), chamferRadius: 0)
    case K_SPHERE, K_SPHERE_WIRES:
        geo = SCNSphere(radius: CGFloat(cmd.w))
    case K_CYLINDER:
        geo = SCNCylinder(radius: CGFloat(cmd.w), height: CGFloat(cmd.h))
    case K_PLANE:
        geo = SCNPlane(width: CGFloat(cmd.w), height: CGFloat(cmd.h))
    case K_GRID:
        return makeGridNode(slices: Int(cmd.w), spacing: Float(cmd.h),
                            color: color)
    default:
        return nil
    }
    guard let g = geo else { return nil }
    let m = SCNMaterial()
    m.lightingModel = .physicallyBased
    m.diffuse.contents = color
    m.roughness.contents = 0.6
    m.metalness.contents = 0.0
    if cmd.kind == K_CUBE_WIRES || cmd.kind == K_SPHERE_WIRES {
        m.fillMode = .lines
    }
    g.materials = [m]

    let node = SCNNode(geometry: g)
    node.position = pos
    return node
}

/// Build a grid as line segments — SceneKit doesn't have a built-in grid,
/// so we construct two sets of parallel SCNGeometry lines.
private func makeGridNode(slices: Int, spacing: Float, color: UIColor) -> SCNNode {
    let root = SCNNode()
    let s = slices < 1 ? 10 : slices
    let half = Float(s) * spacing / 2.0

    var verts: [SCNVector3] = []
    var indices: [Int32] = []
    var idx: Int32 = 0
    for i in 0...s {
        let p = -half + Float(i) * spacing
        verts.append(SCNVector3(p, 0, -half))
        verts.append(SCNVector3(p, 0,  half))
        indices.append(idx); indices.append(idx + 1); idx += 2
        verts.append(SCNVector3(-half, 0, p))
        verts.append(SCNVector3( half, 0, p))
        indices.append(idx); indices.append(idx + 1); idx += 2
    }
    let src = SCNGeometrySource(vertices: verts)
    let data = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
    let el = SCNGeometryElement(data: data, primitiveType: .line,
                                primitiveCount: indices.count / 2,
                                bytesPerIndex: MemoryLayout<Int32>.size)
    let geom = SCNGeometry(sources: [src], elements: [el])
    let mat = SCNMaterial()
    mat.diffuse.contents = color
    mat.isDoubleSided = true
    mat.lightingModel = .constant
    geom.materials = [mat]
    root.geometry = geom
    return root
}

// MARK: - Single-command draw

private func drawOne(ctx: GraphicsContext, cmdPtr: UnsafePointer<DrawCmd>) {
    let cmd = cmdPtr.pointee
    let color = Color(.sRGB,
                      red: cmd.r / 255.0,
                      green: cmd.g / 255.0,
                      blue: cmd.b / 255.0,
                      opacity: cmd.a / 255.0)

    switch cmd.kind {
    case K_RECT:
        ctx.fill(Path(CGRect(x: cmd.x, y: cmd.y, width: cmd.w, height: cmd.h)),
                 with: .color(color))

    case K_RECT_LINES:
        ctx.stroke(Path(CGRect(x: cmd.x, y: cmd.y, width: cmd.w, height: cmd.h)),
                   with: .color(color),
                   lineWidth: cmd.thickness)

    case K_CIRCLE:
        ctx.fill(Path(ellipseIn: CGRect(x: cmd.x - cmd.w, y: cmd.y - cmd.w,
                                        width: cmd.w * 2, height: cmd.w * 2)),
                 with: .color(color))

    case K_CIRCLE_LINES:
        ctx.stroke(Path(ellipseIn: CGRect(x: cmd.x - cmd.w, y: cmd.y - cmd.w,
                                          width: cmd.w * 2, height: cmd.w * 2)),
                   with: .color(color),
                   lineWidth: cmd.thickness)

    case K_LINE:
        var path = Path()
        path.move(to: CGPoint(x: cmd.x, y: cmd.y))
        path.addLine(to: CGPoint(x: cmd.srcX, y: cmd.srcY))
        ctx.stroke(path, with: .color(color), lineWidth: cmd.thickness)

    case K_TRIANGLE:
        var path = Path()
        path.move(to: CGPoint(x: cmd.x, y: cmd.y))
        path.addLine(to: CGPoint(x: cmd.srcX, y: cmd.srcY))
        path.addLine(to: CGPoint(x: cmd.srcW, y: cmd.srcH))
        path.closeSubpath()
        ctx.fill(path, with: .color(color))

    case K_TEXTURE:
        drawTexture(ctx: ctx, cmd: cmd, dstRect: CGRect(x: cmd.x, y: cmd.y, width: cmd.w, height: cmd.h), srcRect: nil)

    case K_TEXTURE_REC:
        let src = CGRect(x: cmd.srcX, y: cmd.srcY, width: cmd.srcW, height: cmd.srcH)
        let dst = CGRect(x: cmd.x, y: cmd.y, width: cmd.w, height: cmd.h)
        drawTexture(ctx: ctx, cmd: cmd, dstRect: dst, srcRect: src)

    case K_TEXTURE_PRO:
        let src = CGRect(x: cmd.srcX, y: cmd.srcY, width: cmd.srcW, height: cmd.srcH)
        let dst = CGRect(x: cmd.x - cmd.ox, y: cmd.y - cmd.oy, width: cmd.w, height: cmd.h)
        if cmd.rot != 0 {
            var nested = ctx
            nested.translateBy(x: cmd.x, y: cmd.y)
            nested.rotate(by: .degrees(cmd.rot))
            nested.translateBy(x: -cmd.ox, y: -cmd.oy)
            drawTexture(ctx: nested, cmd: cmd,
                        dstRect: CGRect(x: 0, y: 0, width: cmd.w, height: cmd.h),
                        srcRect: src)
        } else {
            drawTexture(ctx: ctx, cmd: cmd, dstRect: dst, srcRect: src)
        }

    case K_TEXT:
        let s = cmdTextString(cmdPtr)
        if s.isEmpty { return }
        let font = Font.system(size: cmd.size > 0 ? cmd.size : 14)
        let text = Text(s).font(font).foregroundColor(color)
        ctx.draw(text, at: CGPoint(x: cmd.x, y: cmd.y), anchor: .topLeading)

    default:
        break
    }
}

private func drawTexture(ctx: GraphicsContext, cmd: DrawCmd, dstRect: CGRect, srcRect: CGRect?) {
    guard let full = TextureCache.shared.image(for: cmd.tex) else { return }
    let cg: CGImage
    if let s = srcRect, let cropped = full.cropping(to: s) {
        cg = cropped
    } else {
        cg = full
    }
    let img = Image(cg, scale: 1.0, label: Text(""))
    ctx.draw(img, in: dstRect)
}
