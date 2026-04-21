// BloomWatchApp — SwiftUI @main for bloom-backed watchOS games.
//
// Compiled by Perry when --features watchos-swift-app is on. Owns the process
// entry (@main), spawns the game thread calling _perry_user_main, and renders
// the bloom draw list through a SwiftUI Canvas. Input (Digital Crown + taps)
// feeds back into bloom via the C hooks in lib.rs.

import SwiftUI
import Foundation

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
        guard let url = URL(string: "file://" + path) ?? URL(fileURLWithPath: path) as URL?,
              let provider = CGDataProvider(url: url as CFURL),
              let img = CGImage(pngDataProviderSource: provider,
                                decode: nil,
                                shouldInterpolate: false,
                                intent: .defaultIntent)
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

    let refresh = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                // Re-reading frameTick as the primary invalidation signal.
                _ = frameTick

                // Clear with bloom's requested background.
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

                // Pull this frame's commands.
                let n = Int(bloom_watchos_copy_draw_list(drawBuf.baseAddress!, 4096))
                for i in 0..<n {
                    drawOne(ctx: ctx, cmdPtr: drawBuf.baseAddress!.advanced(by: i))
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
                if f != frameTick { frameTick = f }
            }
        }
        .ignoresSafeArea()
        .background(Color.black)
    }
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
