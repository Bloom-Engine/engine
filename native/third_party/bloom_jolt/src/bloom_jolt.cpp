/*
 * bloom_jolt.cpp — Tier 1 implementation.
 *
 * Handle encoding:
 *   bj_world       → reinterpret_cast<WorldImpl *>
 *   bj_shape       → reinterpret_cast<const Shape *> (intrusive refcounted)
 *   bj_body        → (BodyID.GetIndexAndSequenceNumber() + 1)  — shift by 1
 *                    so Jolt's valid id 0 doesn't collide with BJ_INVALID==0.
 *   bj_constraint  → reinterpret_cast<Constraint *> (intrusive refcounted)
 *
 * Thread safety: the contact listener is invoked from Jolt job threads.
 * We push events into a mutex-guarded queue; user code drains on the main
 * thread via bj_world_pop_contacts.
 */

#include "bloom_jolt.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Body/BodyLockMulti.h>
#include <Jolt/Physics/Body/MotionType.h>
#include <Jolt/Physics/Body/MotionQuality.h>
#include <Jolt/Physics/Body/AllowedDOFs.h>
#include <Jolt/Physics/EActivation.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/CylinderShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/OffsetCenterOfMassShape.h>
#include <Jolt/Physics/Constraints/Constraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/PointConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>
#include <Jolt/Physics/Constraints/SliderConstraint.h>
#include <Jolt/Physics/Constraints/DistanceConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/Vehicle/VehicleConstraint.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleDifferential.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace JPH;

/* ================================================================== */
/*  Anonymous namespace: helper types                                  */
/* ================================================================== */

namespace {

std::atomic<int> g_init_refcount{0};

/* ---- Broadphase layers ---- */
enum : uint8_t {
    BP_NON_MOVING = 0,
    BP_MOVING     = 1,
    BP_COUNT      = 2
};

class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BP_COUNT; }
    BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer layer) const override {
        return layer == (ObjectLayer)BJ_LAYER_NON_MOVING
            ? BroadPhaseLayer(BP_NON_MOVING)
            : BroadPhaseLayer(BP_MOVING);
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(BroadPhaseLayer layer) const override {
        switch ((BroadPhaseLayer::Type)layer) {
            case (BroadPhaseLayer::Type)BP_NON_MOVING: return "NON_MOVING";
            case (BroadPhaseLayer::Type)BP_MOVING:     return "MOVING";
            default:                                    return "INVALID";
        }
    }
#endif
};

class ObjectVsBPFilter final : public ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(ObjectLayer object, BroadPhaseLayer broad) const override {
        if (object == (ObjectLayer)BJ_LAYER_NON_MOVING) {
            return broad == BroadPhaseLayer(BP_MOVING);
        }
        return true;
    }
};

class ObjectLayerPairFilterMatrix final : public ObjectLayerPairFilter {
public:
    ObjectLayerPairFilterMatrix() {
        std::memset(m_matrix, 0xFF, sizeof(m_matrix));
        set(BJ_LAYER_NON_MOVING, BJ_LAYER_NON_MOVING, false);
    }
    bool ShouldCollide(ObjectLayer a, ObjectLayer b) const override {
        if (a >= BJ_MAX_OBJECT_LAYERS || b >= BJ_MAX_OBJECT_LAYERS) return false;
        return (m_matrix[a] & (uint16_t(1) << b)) != 0;
    }
    void set(uint32_t a, uint32_t b, bool collides) {
        if (a >= BJ_MAX_OBJECT_LAYERS || b >= BJ_MAX_OBJECT_LAYERS) return;
        if (collides) {
            m_matrix[a] |= uint16_t(1) << b;
            m_matrix[b] |= uint16_t(1) << a;
        } else {
            m_matrix[a] &= ~(uint16_t(1) << b);
            m_matrix[b] &= ~(uint16_t(1) << a);
        }
    }
    bool get(uint32_t a, uint32_t b) const {
        if (a >= BJ_MAX_OBJECT_LAYERS || b >= BJ_MAX_OBJECT_LAYERS) return false;
        return (m_matrix[a] & (uint16_t(1) << b)) != 0;
    }
private:
    uint16_t m_matrix[BJ_MAX_OBJECT_LAYERS];
};

/* ---- Filter for queries that respect a caller-provided layer mask ---- */

class LayerMaskObjectFilter final : public ObjectLayerFilter {
public:
    explicit LayerMaskObjectFilter(uint32_t mask) : m_mask(mask) {}
    bool ShouldCollide(ObjectLayer layer) const override {
        if (layer >= 32) return true;              /* layers 16-31 not yet partitioned; pass through */
        return (m_mask & (uint32_t(1) << layer)) != 0;
    }
private:
    uint32_t m_mask;
};

class LayerMaskBPFilter final : public BroadPhaseLayerFilter {
public:
    explicit LayerMaskBPFilter(uint32_t mask) : m_mask(mask) {}
    bool ShouldCollide(BroadPhaseLayer layer) const override {
        /* BP_NON_MOVING contains object layer 0; BP_MOVING contains object layers 1..15. */
        uint8_t bp = (BroadPhaseLayer::Type)layer;
        if (bp == BP_NON_MOVING) return (m_mask & 0x1u) != 0;
        if (bp == BP_MOVING)     return (m_mask & 0xFFFEu) != 0;
        return true;
    }
private:
    uint32_t m_mask;
};

/* ---- Contact listener → drainable event queue ---- */

struct WorldImpl; /* forward */

class ContactQueue final : public ContactListener {
public:
    void OnContactAdded(const Body &body1, const Body &body2,
                        const ContactManifold &manifold,
                        ContactSettings &settings) override
    {
        push(BJ_CONTACT_ADDED, body1, body2, &manifold, &settings);
    }
    void OnContactPersisted(const Body &body1, const Body &body2,
                            const ContactManifold &manifold,
                            ContactSettings &settings) override
    {
        push(BJ_CONTACT_PERSISTED, body1, body2, &manifold, &settings);
    }
    void OnContactRemoved(const SubShapeIDPair &pair) override
    {
        bj_contact c{};
        c.event  = BJ_CONTACT_REMOVED;
        c.body_a = encode_body_id(pair.GetBody1ID());
        c.body_b = encode_body_id(pair.GetBody2ID());
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.push_back(c);
    }

    uint32_t count() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return (uint32_t)m_events.size();
    }

    uint32_t drain(bj_contact *out, uint32_t max_out) {
        if (!out || max_out == 0) return 0;
        std::lock_guard<std::mutex> lock(m_mutex);
        uint32_t n = std::min<uint32_t>(max_out, (uint32_t)m_events.size());
        for (uint32_t i = 0; i < n; ++i) out[i] = m_events[i];
        m_events.erase(m_events.begin(), m_events.begin() + n);
        return n;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.clear();
    }

private:
    static bj_vec3 from_vec3(Vec3 v) { return { v.GetX(), v.GetY(), v.GetZ() }; }
    static bj_body encode_body_id(BodyID id) {
        uint32_t raw = id.GetIndexAndSequenceNumber();
        return raw == BodyID::cInvalidBodyID ? BJ_INVALID : (bj_body)(uint64_t(raw) + 1u);
    }
    void push(bj_contact_event ev, const Body &b1, const Body &b2,
              const ContactManifold *m, const ContactSettings *s)
    {
        bj_contact c{};
        c.event  = ev;
        c.body_a = encode_body_id(b1.GetID());
        c.body_b = encode_body_id(b2.GetID());
        if (m) {
            Vec3 base(m->mBaseOffset.GetX(), m->mBaseOffset.GetY(), m->mBaseOffset.GetZ());
            Vec3 p1 = m->mRelativeContactPointsOn1.empty()
                ? base
                : base + m->mRelativeContactPointsOn1[0];
            Vec3 p2 = m->mRelativeContactPointsOn2.empty()
                ? base
                : base + m->mRelativeContactPointsOn2[0];
            c.point_a           = from_vec3(p1);
            c.point_b           = from_vec3(p2);
            c.normal            = from_vec3(m->mWorldSpaceNormal);
            c.penetration_depth = m->mPenetrationDepth;
        }
        if (s) {
            c.combined_friction    = s->mCombinedFriction;
            c.combined_restitution = s->mCombinedRestitution;
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        m_events.push_back(c);
    }

    std::mutex              m_mutex;
    std::vector<bj_contact> m_events;
};

/* ---- World ---- */

struct WorldImpl {
    TempAllocatorImpl           *temp_alloc = nullptr;
    JobSystemThreadPool         *jobs       = nullptr;
    BPLayerInterfaceImpl         bp_layer_interface;
    ObjectVsBPFilter             object_vs_bp_filter;
    ObjectLayerPairFilterMatrix  layer_pair_filter;
    PhysicsSystem                system;
    ContactQueue                 contacts;
};

/* ---- Encode / decode ---- */

inline WorldImpl *as_world(bj_world w)  { return reinterpret_cast<WorldImpl *>(w); }
inline bj_world   as_handle(WorldImpl *w) { return reinterpret_cast<bj_world>(w); }

inline bj_body encode_body_id(BodyID id) {
    uint32_t raw = id.GetIndexAndSequenceNumber();
    return raw == BodyID::cInvalidBodyID ? BJ_INVALID : (bj_body)(uint64_t(raw) + 1u);
}
inline BodyID decode_body_id(bj_body b) {
    if (b == BJ_INVALID) return BodyID();
    return BodyID((uint32_t)(b - 1u));
}

inline bj_shape encode_shape(const Shape *s) {
    if (!s) return BJ_INVALID;
    s->AddRef();  /* handle owns one ref */
    return reinterpret_cast<bj_shape>(s);
}
inline const Shape *decode_shape(bj_shape h) {
    return reinterpret_cast<const Shape *>(h);
}

inline bj_constraint encode_constraint(Constraint *c) {
    if (!c) return BJ_INVALID;
    c->AddRef();
    return reinterpret_cast<bj_constraint>(c);
}
inline Constraint *decode_constraint(bj_constraint h) {
    return reinterpret_cast<Constraint *>(h);
}

/* ---- Vec3/Quat conversion ---- */

inline Vec3    to_jph(bj_vec3 v) { return Vec3(v.x, v.y, v.z); }
inline Quat    to_jph(bj_quat q) { return Quat(q.x, q.y, q.z, q.w); }
inline bj_vec3 from_jph(Vec3 v) { return { v.GetX(), v.GetY(), v.GetZ() }; }
inline bj_quat from_jph(Quat q) { return { q.GetX(), q.GetY(), q.GetZ(), q.GetW() }; }
/* In single-precision builds RVec3 == Vec3, so to_jph/from_jph cover both. */

inline EMotionType to_motion(bj_motion_type t) {
    switch (t) {
        case BJ_MOTION_STATIC:    return EMotionType::Static;
        case BJ_MOTION_KINEMATIC: return EMotionType::Kinematic;
        case BJ_MOTION_DYNAMIC:   return EMotionType::Dynamic;
    }
    return EMotionType::Dynamic;
}
inline bj_motion_type from_motion(EMotionType t) {
    switch (t) {
        case EMotionType::Static:    return BJ_MOTION_STATIC;
        case EMotionType::Kinematic: return BJ_MOTION_KINEMATIC;
        case EMotionType::Dynamic:   return BJ_MOTION_DYNAMIC;
    }
    return BJ_MOTION_DYNAMIC;
}
inline EActivation to_activation(bj_activation a) {
    return a == BJ_ACTIVATE ? EActivation::Activate : EActivation::DontActivate;
}

/* Run `build` under a multi-body write lock and add the resulting constraint
 * to the world. If body_b is BJ_INVALID, the world-fixed body is used. */
template <typename Build>
bj_constraint make_constraint(WorldImpl *w, const bj_constraint_anchors *a, Build &&build) {
    BodyID id1 = decode_body_id(a->body_a);
    BodyID id2 = (a->body_b != BJ_INVALID) ? decode_body_id(a->body_b) : BodyID();
    BodyID ids[2] = { id1, id2 };
    const int count = (a->body_b != BJ_INVALID) ? 2 : 1;
    BodyLockMultiWrite lock(w->system.GetBodyLockInterface(), ids, count);
    Body *b1 = lock.GetBody(0);
    Body *b2 = (count == 2) ? lock.GetBody(1) : &Body::sFixedToWorld;
    if (!b1 || !b2) return BJ_INVALID;
    TwoBodyConstraint *c = build(*b1, *b2);
    if (!c) return BJ_INVALID;
    w->system.AddConstraint(c);
    return encode_constraint(c);
}

} /* anonymous namespace */

/* ================================================================== */
/*                           IMPLEMENTATIONS                           */
/* ================================================================== */

extern "C" {

/* ---- Global ---- */

bj_result bj_global_init(void) {
    if (g_init_refcount.fetch_add(1) == 0) {
        RegisterDefaultAllocator();
        Factory::sInstance = new Factory();
        RegisterTypes();
    }
    return BJ_OK;
}

void bj_global_shutdown(void) {
    int prev = g_init_refcount.fetch_sub(1);
    if (prev == 1) {
        UnregisterTypes();
        delete Factory::sInstance;
        Factory::sInstance = nullptr;
    }
}

const char *bj_version_string(void) {
    return "Jolt 5.5.0 / bloom_jolt 0.1";
}

/* ---- World ---- */

bj_world bj_world_create(const bj_world_desc *desc) {
    if (!desc) return BJ_INVALID;
    if (g_init_refcount.load() == 0) return BJ_INVALID;

    const uint32_t max_bodies =
        desc->max_bodies ? desc->max_bodies : 65536u;
    const uint32_t max_body_pairs =
        desc->max_body_pairs ? desc->max_body_pairs : 65536u;
    const uint32_t max_contact_constraints =
        desc->max_contact_constraints ? desc->max_contact_constraints : 10240u;
    const int num_threads = desc->num_threads
        ? (int)desc->num_threads
        : std::max(1, (int)std::thread::hardware_concurrency() - 1);
    const uint32_t temp_alloc_bytes =
        desc->temp_allocator_bytes ? desc->temp_allocator_bytes : (10u * 1024u * 1024u);

    auto *w = new WorldImpl();
    w->temp_alloc = new TempAllocatorImpl(temp_alloc_bytes);
    w->jobs       = new JobSystemThreadPool(cMaxPhysicsJobs, cMaxPhysicsBarriers, num_threads);

    w->system.Init(
        max_bodies, /*numBodyMutexes=*/0,
        max_body_pairs, max_contact_constraints,
        w->bp_layer_interface, w->object_vs_bp_filter, w->layer_pair_filter
    );
    w->system.SetGravity(to_jph(desc->gravity));
    w->system.SetContactListener(&w->contacts);
    return as_handle(w);
}

void bj_world_destroy(bj_world world) {
    WorldImpl *w = as_world(world);
    if (!w) return;
    delete w->jobs;
    delete w->temp_alloc;
    delete w;
}

void bj_world_set_gravity(bj_world world, bj_vec3 gravity) {
    if (auto *w = as_world(world)) w->system.SetGravity(to_jph(gravity));
}

void bj_world_get_gravity(bj_world world, bj_vec3 *out) {
    if (!out) return;
    if (auto *w = as_world(world)) *out = from_jph(w->system.GetGravity());
    else *out = { 0.0f, 0.0f, 0.0f };
}

void bj_world_optimize_broadphase(bj_world world) {
    if (auto *w = as_world(world)) w->system.OptimizeBroadPhase();
}

bj_result bj_world_step(bj_world world, float delta_time, uint32_t collision_steps) {
    WorldImpl *w = as_world(world);
    if (!w) return BJ_ERR_INVALID_HANDLE;
    const int steps = collision_steps ? (int)collision_steps : 1;
    EPhysicsUpdateError err = w->system.Update(delta_time, steps, w->temp_alloc, w->jobs);
    return err == EPhysicsUpdateError::None ? BJ_OK : BJ_ERR_OUT_OF_MEMORY;
}

void bj_world_set_layer_collides(bj_world world, uint32_t a, uint32_t b, uint8_t collides) {
    if (auto *w = as_world(world)) w->layer_pair_filter.set(a, b, collides != 0);
}

uint8_t bj_world_get_layer_collides(bj_world world, uint32_t a, uint32_t b) {
    if (auto *w = as_world(world)) return w->layer_pair_filter.get(a, b) ? 1 : 0;
    return 0;
}

uint32_t bj_world_body_count(bj_world world) {
    if (auto *w = as_world(world)) return w->system.GetNumBodies();
    return 0;
}

uint32_t bj_world_active_body_count(bj_world world) {
    if (auto *w = as_world(world)) return w->system.GetNumActiveBodies(EBodyType::RigidBody);
    return 0;
}

/* ================================================================== */
/*  Shapes                                                             */
/* ================================================================== */

bj_shape bj_shape_box(bj_vec3 half_extents, float convex_radius) {
    BoxShapeSettings settings(to_jph(half_extents), convex_radius);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_sphere(float radius) {
    SphereShapeSettings settings(radius);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_capsule(float half_height, float radius) {
    CapsuleShapeSettings settings(half_height, radius);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_cylinder(float half_height, float radius, float convex_radius) {
    CylinderShapeSettings settings(half_height, radius, convex_radius);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_convex_hull(const bj_vec3 *points, uint32_t count, float convex_radius) {
    if (!points || count < 3) return BJ_INVALID;
    Array<Vec3> pts;
    pts.reserve(count);
    for (uint32_t i = 0; i < count; ++i) pts.push_back(to_jph(points[i]));
    ConvexHullShapeSettings settings(pts, convex_radius);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_mesh(const bj_vec3 *vertices, uint32_t vertex_count,
                       const uint32_t *indices, uint32_t triangle_count)
{
    if (!vertices || !indices || vertex_count == 0 || triangle_count == 0) return BJ_INVALID;
    VertexList v_list;
    v_list.reserve(vertex_count);
    for (uint32_t i = 0; i < vertex_count; ++i) {
        v_list.push_back(Float3(vertices[i].x, vertices[i].y, vertices[i].z));
    }
    IndexedTriangleList t_list;
    t_list.reserve(triangle_count);
    for (uint32_t i = 0; i < triangle_count; ++i) {
        t_list.push_back(IndexedTriangle(indices[i*3+0], indices[i*3+1], indices[i*3+2], 0));
    }
    MeshShapeSettings settings(std::move(v_list), std::move(t_list));
    settings.Sanitize();
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_heightfield(const float *samples, uint32_t sample_count,
                              bj_vec3 offset, bj_vec3 scale, uint32_t block_size)
{
    if (!samples || sample_count < 2) return BJ_INVALID;
    HeightFieldShapeSettings settings(
        samples,
        to_jph(offset),
        to_jph(scale),
        sample_count
    );
    settings.mBlockSize = block_size ? block_size : 4u;
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_compound_static(const bj_shape *shapes,
                                   const bj_transform *local_transforms,
                                   uint32_t count)
{
    if (!shapes || !local_transforms || count == 0) return BJ_INVALID;
    StaticCompoundShapeSettings settings;
    for (uint32_t i = 0; i < count; ++i) {
        const Shape *child = decode_shape(shapes[i]);
        if (!child) return BJ_INVALID;
        settings.AddShape(
            to_jph(local_transforms[i].position),
            to_jph(local_transforms[i].rotation),
            child
        );
    }
    /* StaticCompoundShape::Create needs a temp allocator + job system for parallel BVH build.
     * For simplicity we use the default (nullptr = single-threaded build). */
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_scaled(bj_shape base, bj_vec3 scale) {
    const Shape *inner = decode_shape(base);
    if (!inner) return BJ_INVALID;
    ScaledShapeSettings settings(inner, to_jph(scale));
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

bj_shape bj_shape_offset_com(bj_shape base, bj_vec3 offset) {
    const Shape *inner = decode_shape(base);
    if (!inner) return BJ_INVALID;
    OffsetCenterOfMassShapeSettings settings(to_jph(offset), inner);
    auto result = settings.Create();
    return result.IsValid() ? encode_shape(result.Get().GetPtr()) : BJ_INVALID;
}

void bj_shape_add_ref(bj_shape h) { if (auto *s = decode_shape(h)) s->AddRef(); }
void bj_shape_release(bj_shape h) { if (auto *s = decode_shape(h)) s->Release(); }

void bj_shape_get_local_bounds(bj_shape h, bj_vec3 *out_min, bj_vec3 *out_max) {
    if (auto *s = decode_shape(h)) {
        AABox b = s->GetLocalBounds();
        if (out_min) *out_min = from_jph(b.mMin);
        if (out_max) *out_max = from_jph(b.mMax);
    } else {
        if (out_min) *out_min = { 0, 0, 0 };
        if (out_max) *out_max = { 0, 0, 0 };
    }
}

float bj_shape_get_volume(bj_shape h) {
    if (auto *s = decode_shape(h)) return s->GetVolume();
    return 0.0f;
}

/* ================================================================== */
/*  Bodies                                                             */
/* ================================================================== */

bj_body bj_body_create(bj_world world, bj_shape shape, const bj_body_desc *desc) {
    WorldImpl *w = as_world(world);
    if (!w || !desc) return BJ_INVALID;
    const Shape *s = decode_shape(shape);
    if (!s) return BJ_INVALID;

    BodyCreationSettings settings(
        s,
        to_jph(desc->position),
        to_jph(desc->rotation),
        to_motion(desc->motion_type),
        (ObjectLayer)desc->object_layer
    );
    settings.mLinearVelocity  = to_jph(desc->linear_velocity);
    settings.mAngularVelocity = to_jph(desc->angular_velocity);
    settings.mGravityFactor   = desc->gravity_factor;
    settings.mLinearDamping   = desc->linear_damping;
    settings.mAngularDamping  = desc->angular_damping;
    settings.mFriction        = desc->friction;
    settings.mRestitution     = desc->restitution;
    settings.mIsSensor        = desc->is_sensor != 0;
    settings.mAllowSleeping   = desc->allow_sleeping != 0;
    settings.mMotionQuality   =
        desc->use_ccd ? EMotionQuality::LinearCast : EMotionQuality::Discrete;
    settings.mUserData        = desc->user_data;

    if (desc->mass_override > 0.0f) {
        settings.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
        settings.mMassPropertiesOverride.mMass = desc->mass_override;
        Vec3 inertia_diag = to_jph(desc->inertia_diag_override);
        if (inertia_diag.LengthSq() > 0.0f) {
            settings.mMassPropertiesOverride.mInertia =
                Mat44::sScale(inertia_diag);
        } else {
            /* Auto-compute inertia from shape but scale by the mass ratio. */
            settings.mOverrideMassProperties =
                EOverrideMassProperties::CalculateInertia;
            settings.mMassPropertiesOverride.mMass = desc->mass_override;
        }
    }

    BodyInterface &bi = w->system.GetBodyInterface();
    EActivation act = desc->start_awake ? EActivation::Activate : EActivation::DontActivate;
    BodyID id = bi.CreateAndAddBody(settings, act);
    return encode_body_id(id);
}

void bj_body_destroy(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyInterface &bi = w->system.GetBodyInterface();
    BodyID id = decode_body_id(body);
    if (bi.IsAdded(id)) bi.RemoveBody(id);
    bi.DestroyBody(id);
}

void bj_body_activate(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().ActivateBody(decode_body_id(body));
}

void bj_body_deactivate(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().DeactivateBody(decode_body_id(body));
}

uint8_t bj_body_is_active(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0;
    return w->system.GetBodyInterface().IsActive(decode_body_id(body)) ? 1 : 0;
}

uint8_t bj_body_is_valid(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0;
    return w->system.GetBodyInterface().IsAdded(decode_body_id(body)) ? 1 : 0;
}

void bj_body_get_transform(bj_world world, bj_body body, bj_transform *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) {
        *out = { { 0, 0, 0 }, { 0, 0, 0, 1 } };
        return;
    }
    RVec3 p;
    Quat q;
    w->system.GetBodyInterface().GetPositionAndRotation(decode_body_id(body), p, q);
    out->position = from_jph(p);
    out->rotation = from_jph(q);
}

void bj_body_set_transform(bj_world world, bj_body body, const bj_transform *xform, bj_activation act) {
    WorldImpl *w = as_world(world);
    if (!w || !xform || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetPositionAndRotation(
        decode_body_id(body),
        to_jph(xform->position),
        to_jph(xform->rotation),
        to_activation(act)
    );
}

void bj_body_get_position(bj_world world, bj_body body, bj_vec3 *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(w->system.GetBodyInterface().GetPosition(decode_body_id(body)));
}

void bj_body_get_rotation(bj_world world, bj_body body, bj_quat *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) { *out = { 0, 0, 0, 1 }; return; }
    *out = from_jph(w->system.GetBodyInterface().GetRotation(decode_body_id(body)));
}

void bj_body_set_position(bj_world world, bj_body body, bj_vec3 pos, bj_activation act) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetPosition(decode_body_id(body), to_jph(pos), to_activation(act));
}

void bj_body_set_rotation(bj_world world, bj_body body, bj_quat rot, bj_activation act) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetRotation(decode_body_id(body), to_jph(rot), to_activation(act));
}

void bj_body_move_kinematic(bj_world world, bj_body body, const bj_transform *target, float dt) {
    WorldImpl *w = as_world(world);
    if (!w || !target || body == BJ_INVALID || dt <= 0.0f) return;
    w->system.GetBodyInterface().MoveKinematic(
        decode_body_id(body), to_jph(target->position), to_jph(target->rotation), dt
    );
}

void bj_body_get_linear_velocity(bj_world world, bj_body body, bj_vec3 *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(w->system.GetBodyInterface().GetLinearVelocity(decode_body_id(body)));
}

void bj_body_get_angular_velocity(bj_world world, bj_body body, bj_vec3 *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(w->system.GetBodyInterface().GetAngularVelocity(decode_body_id(body)));
}

void bj_body_set_linear_velocity(bj_world world, bj_body body, bj_vec3 v) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetLinearVelocity(decode_body_id(body), to_jph(v));
}

void bj_body_set_angular_velocity(bj_world world, bj_body body, bj_vec3 v) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetAngularVelocity(decode_body_id(body), to_jph(v));
}

void bj_body_get_point_velocity(bj_world world, bj_body body, bj_vec3 world_point, bj_vec3 *out) {
    if (!out) return;
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(w->system.GetBodyInterface().GetPointVelocity(decode_body_id(body), to_jph(world_point)));
}

void bj_body_add_force(bj_world world, bj_body body, bj_vec3 f) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddForce(decode_body_id(body), to_jph(f));
}

void bj_body_add_impulse(bj_world world, bj_body body, bj_vec3 i) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddImpulse(decode_body_id(body), to_jph(i));
}

void bj_body_add_torque(bj_world world, bj_body body, bj_vec3 t) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddTorque(decode_body_id(body), to_jph(t));
}

void bj_body_add_angular_impulse(bj_world world, bj_body body, bj_vec3 i) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddAngularImpulse(decode_body_id(body), to_jph(i));
}

void bj_body_add_force_at(bj_world world, bj_body body, bj_vec3 f, bj_vec3 p) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddForce(decode_body_id(body), to_jph(f), to_jph(p));
}

void bj_body_add_impulse_at(bj_world world, bj_body body, bj_vec3 i, bj_vec3 p) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().AddImpulse(decode_body_id(body), to_jph(i), to_jph(p));
}

void bj_body_set_friction(bj_world world, bj_body body, float f) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetFriction(decode_body_id(body), f);
}

void bj_body_set_restitution(bj_world world, bj_body body, float r) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetRestitution(decode_body_id(body), r);
}

void bj_body_set_linear_damping(bj_world world, bj_body body, float d) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) {
        if (MotionProperties *mp = lock.GetBody().GetMotionPropertiesUnchecked()) {
            mp->SetLinearDamping(d);
        }
    }
}

void bj_body_set_angular_damping(bj_world world, bj_body body, float d) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) {
        if (MotionProperties *mp = lock.GetBody().GetMotionPropertiesUnchecked()) {
            mp->SetAngularDamping(d);
        }
    }
}

void bj_body_set_gravity_factor(bj_world world, bj_body body, float factor) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetGravityFactor(decode_body_id(body), factor);
}

void bj_body_set_ccd(bj_world world, bj_body body, uint8_t enabled) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetMotionQuality(
        decode_body_id(body),
        enabled ? EMotionQuality::LinearCast : EMotionQuality::Discrete
    );
}

void bj_body_set_motion_type(bj_world world, bj_body body, bj_motion_type t, bj_activation act) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetMotionType(decode_body_id(body), to_motion(t), to_activation(act));
}

void bj_body_set_object_layer(bj_world world, bj_body body, uint32_t layer) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetObjectLayer(decode_body_id(body), (ObjectLayer)layer);
}

void bj_body_set_is_sensor(bj_world world, bj_body body, uint8_t enabled) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) lock.GetBody().SetIsSensor(enabled != 0);
}

void bj_body_set_allow_sleeping(bj_world world, bj_body body, uint8_t enabled) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) lock.GetBody().SetAllowSleeping(enabled != 0);
}

void bj_body_set_shape(bj_world world, bj_body body, bj_shape shape, uint8_t update_mass, bj_activation act) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    const Shape *s = decode_shape(shape);
    if (!s) return;
    w->system.GetBodyInterface().SetShape(decode_body_id(body), s, update_mass != 0, to_activation(act));
}

void bj_body_lock_rotation_axes(bj_world world, bj_body body, uint8_t lx, uint8_t ly, uint8_t lz) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    /* AllowedDOFs: bit 0-2 = translation X/Y/Z, bit 3-5 = rotation X/Y/Z. */
    uint8_t dofs = (uint8_t)EAllowedDOFs::All;
    if (lx) dofs &= ~(uint8_t)EAllowedDOFs::RotationX;
    if (ly) dofs &= ~(uint8_t)EAllowedDOFs::RotationY;
    if (lz) dofs &= ~(uint8_t)EAllowedDOFs::RotationZ;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) {
        if (MotionProperties *mp = lock.GetBody().GetMotionPropertiesUnchecked()) {
            /* Note: we preserve translation DOFs — lock_translation_axes is separate. */
            EAllowedDOFs existing = mp->GetAllowedDOFs();
            uint8_t trans_bits = (uint8_t)existing & 0x07u;
            mp->SetInverseInertia(
                mp->GetInverseInertiaDiagonal(),
                mp->GetInertiaRotation()
            );  /* keep inertia; Jolt has no direct AllowedDOFs setter post-create */
            /* Fall-back: zero the inverse inertia on locked axes. */
            Vec3 inv = mp->GetInverseInertiaDiagonal();
            if (lx) inv.SetX(0.0f);
            if (ly) inv.SetY(0.0f);
            if (lz) inv.SetZ(0.0f);
            mp->SetInverseInertia(inv, mp->GetInertiaRotation());
            (void)trans_bits;
            (void)dofs;
        }
    }
}

void bj_body_lock_translation_axes(bj_world world, bj_body body, uint8_t lx, uint8_t ly, uint8_t lz) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (lock.Succeeded()) {
        if (MotionProperties *mp = lock.GetBody().GetMotionPropertiesUnchecked()) {
            /* Zero the inverse mass on locked translation axes via a motion-type-like hack:
             * we cannot change AllowedDOFs post-create in 5.5, so we clamp velocity each step.
             * For now we simply zero the current linear velocity component; full fidelity
             * requires re-creating the body. Tier 2 will add a proper API. */
            Vec3 v = lock.GetBody().GetLinearVelocity();
            if (lx) v.SetX(0.0f);
            if (ly) v.SetY(0.0f);
            if (lz) v.SetZ(0.0f);
            lock.GetBody().SetLinearVelocity(v);
        }
    }
}

float bj_body_get_mass(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0.0f;
    BodyLockRead lock(w->system.GetBodyLockInterface(), decode_body_id(body));
    if (!lock.Succeeded()) return 0.0f;
    const MotionProperties *mp = lock.GetBody().GetMotionPropertiesUnchecked();
    if (!mp) return 0.0f;
    float inv = mp->GetInverseMass();
    return inv > 0.0f ? 1.0f / inv : 0.0f;
}

float bj_body_get_friction(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0.0f;
    return w->system.GetBodyInterface().GetFriction(decode_body_id(body));
}

float bj_body_get_restitution(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0.0f;
    return w->system.GetBodyInterface().GetRestitution(decode_body_id(body));
}

uint32_t bj_body_get_object_layer(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0;
    return (uint32_t)w->system.GetBodyInterface().GetObjectLayer(decode_body_id(body));
}

void bj_body_set_user_data(bj_world world, bj_body body, uint64_t user_data) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return;
    w->system.GetBodyInterface().SetUserData(decode_body_id(body), user_data);
}

uint64_t bj_body_get_user_data(bj_world world, bj_body body) {
    WorldImpl *w = as_world(world);
    if (!w || body == BJ_INVALID) return 0;
    return w->system.GetBodyInterface().GetUserData(decode_body_id(body));
}

uint32_t bj_world_sync_transforms(bj_world world, const bj_body *bodies, uint32_t count, bj_transform *out) {
    WorldImpl *w = as_world(world);
    if (!w || !bodies || !out || count == 0) return 0;
    const BodyLockInterface &lock_if = w->system.GetBodyLockInterface();
    uint32_t valid = 0;
    for (uint32_t i = 0; i < count; ++i) {
        out[i] = { { 0, 0, 0 }, { 0, 0, 0, 1 } };
        if (bodies[i] == BJ_INVALID) continue;
        BodyLockRead lock(lock_if, decode_body_id(bodies[i]));
        if (!lock.Succeeded()) continue;
        RVec3 p = lock.GetBody().GetPosition();
        Quat  q = lock.GetBody().GetRotation();
        out[i].position = from_jph(p);
        out[i].rotation = from_jph(q);
        ++valid;
    }
    return valid;
}

/* ================================================================== */
/*  Queries                                                            */
/* ================================================================== */

uint8_t bj_query_raycast_closest(bj_world world,
                                 bj_vec3 origin, bj_vec3 direction, float max_distance,
                                 uint32_t layer_mask, bj_ray_hit *out_hit)
{
    WorldImpl *w = as_world(world);
    if (!w || !out_hit) return 0;

    Vec3 dir = to_jph(direction);
    if (dir.LengthSq() == 0.0f) return 0;
    dir = dir.Normalized() * max_distance;

    RRayCast ray(to_jph(origin), dir);
    RayCastResult hit;
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);

    bool any = w->system.GetNarrowPhaseQuery().CastRay(ray, hit, bp_filter, obj_filter);
    if (!any) return 0;

    Vec3 point = Vec3(ray.mOrigin) + ray.mDirection * hit.mFraction;
    out_hit->body         = encode_body_id(hit.mBodyID);
    out_hit->point        = from_jph(point);
    /* Compute normal via body lock */
    {
        BodyLockRead lock(w->system.GetBodyLockInterface(), hit.mBodyID);
        if (lock.Succeeded()) {
            Vec3 n = lock.GetBody().GetWorldSpaceSurfaceNormal(hit.mSubShapeID2, RVec3(point));
            out_hit->normal = from_jph(n);
        } else {
            out_hit->normal = { 0.0f, 1.0f, 0.0f };
        }
    }
    out_hit->fraction     = hit.mFraction;
    out_hit->sub_shape_id = hit.mSubShapeID2.GetValue();
    return 1;
}

uint32_t bj_query_raycast_all(bj_world world,
                              bj_vec3 origin, bj_vec3 direction, float max_distance,
                              uint32_t layer_mask,
                              bj_ray_hit *out_hits, uint32_t max_hits)
{
    WorldImpl *w = as_world(world);
    if (!w || !out_hits || max_hits == 0) return 0;

    Vec3 dir = to_jph(direction);
    if (dir.LengthSq() == 0.0f) return 0;
    dir = dir.Normalized() * max_distance;

    RRayCast ray(to_jph(origin), dir);
    RayCastSettings settings;
    settings.mBackFaceModeTriangles = EBackFaceMode::CollideWithBackFaces;
    AllHitCollisionCollector<CastRayCollector> collector;
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);

    w->system.GetNarrowPhaseQuery().CastRay(ray, settings, collector, bp_filter, obj_filter);
    collector.Sort();

    uint32_t n = std::min<uint32_t>(max_hits, (uint32_t)collector.mHits.size());
    for (uint32_t i = 0; i < n; ++i) {
        const RayCastResult &h = collector.mHits[i];
        Vec3 point = Vec3(ray.mOrigin) + ray.mDirection * h.mFraction;
        out_hits[i].body         = encode_body_id(h.mBodyID);
        out_hits[i].point        = from_jph(point);
        BodyLockRead lock(w->system.GetBodyLockInterface(), h.mBodyID);
        if (lock.Succeeded()) {
            Vec3 nrm = lock.GetBody().GetWorldSpaceSurfaceNormal(h.mSubShapeID2, RVec3(point));
            out_hits[i].normal = from_jph(nrm);
        } else {
            out_hits[i].normal = { 0, 1, 0 };
        }
        out_hits[i].fraction     = h.mFraction;
        out_hits[i].sub_shape_id = h.mSubShapeID2.GetValue();
    }
    return n;
}

uint8_t bj_query_shape_cast_closest(bj_world world, bj_shape shape,
                                    const bj_transform *start, bj_vec3 direction,
                                    uint32_t layer_mask, bj_ray_hit *out_hit)
{
    WorldImpl *w = as_world(world);
    if (!w || !start || !out_hit) return 0;
    const Shape *s = decode_shape(shape);
    if (!s) return 0;

    RShapeCast cast = RShapeCast::sFromWorldTransform(
        s, Vec3::sOne(),
        RMat44::sRotationTranslation(to_jph(start->rotation), to_jph(start->position)),
        to_jph(direction)
    );
    ClosestHitCollisionCollector<CastShapeCollector> collector;
    ShapeCastSettings settings;
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);

    w->system.GetNarrowPhaseQuery().CastShape(cast, settings, cast.mCenterOfMassStart.GetTranslation(), collector, bp_filter, obj_filter);
    if (!collector.HadHit()) return 0;

    const ShapeCastResult &h = collector.mHit;
    out_hit->body         = encode_body_id(h.mBodyID2);
    out_hit->point        = from_jph(h.mContactPointOn2);
    out_hit->normal       = from_jph(-h.mPenetrationAxis.Normalized());
    out_hit->fraction     = h.mFraction;
    out_hit->sub_shape_id = h.mSubShapeID2.GetValue();
    return 1;
}

namespace {
class BodyCollector final : public CollideShapeBodyCollector {
public:
    BodyCollector(bj_body *out, uint32_t max_out) : m_out(out), m_max(max_out) {}
    void AddHit(const BodyID &id) override {
        if (m_count < m_max) {
            m_out[m_count++] = encode_body_id(id);
        }
    }
    uint32_t count() const { return m_count; }
private:
    bj_body *m_out;
    uint32_t m_max;
    uint32_t m_count = 0;
};
} /* namespace */

uint32_t bj_query_overlap_sphere(bj_world world, bj_vec3 center, float radius,
                                 uint32_t layer_mask, bj_body *out_bodies, uint32_t max_results)
{
    WorldImpl *w = as_world(world);
    if (!w || !out_bodies || max_results == 0) return 0;
    BodyCollector collector(out_bodies, max_results);
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);
    w->system.GetBroadPhaseQuery().CollideSphere(to_jph(center), radius, collector, bp_filter, obj_filter);
    return collector.count();
}

uint32_t bj_query_overlap_box(bj_world world, const bj_transform *xform, bj_vec3 half_extents,
                              uint32_t layer_mask, bj_body *out_bodies, uint32_t max_results)
{
    WorldImpl *w = as_world(world);
    if (!w || !xform || !out_bodies || max_results == 0) return 0;
    /* Approximate by AABB around the rotated box. */
    Vec3 he = to_jph(half_extents);
    Mat44 rot = Mat44::sRotation(to_jph(xform->rotation));
    Vec3 r0 = rot * Vec3(he.GetX(), 0, 0);
    Vec3 r1 = rot * Vec3(0, he.GetY(), 0);
    Vec3 r2 = rot * Vec3(0, 0, he.GetZ());
    Vec3 extent = r0.Abs() + r1.Abs() + r2.Abs();
    AABox box(to_jph(xform->position) - extent, to_jph(xform->position) + extent);
    BodyCollector collector(out_bodies, max_results);
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);
    w->system.GetBroadPhaseQuery().CollideAABox(box, collector, bp_filter, obj_filter);
    return collector.count();
}

uint32_t bj_query_overlap_point(bj_world world, bj_vec3 point,
                                uint32_t layer_mask, bj_body *out_bodies, uint32_t max_results)
{
    WorldImpl *w = as_world(world);
    if (!w || !out_bodies || max_results == 0) return 0;
    BodyCollector collector(out_bodies, max_results);
    LayerMaskBPFilter bp_filter(layer_mask);
    LayerMaskObjectFilter obj_filter(layer_mask);
    w->system.GetBroadPhaseQuery().CollidePoint(to_jph(point), collector, bp_filter, obj_filter);
    return collector.count();
}

/* ================================================================== */
/*  Constraints                                                        */
/* ================================================================== */

bj_constraint bj_constraint_fixed(bj_world world, const bj_constraint_anchors *a) {
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        FixedConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPoint1 = to_jph(a->anchor_a);
        s.mPoint2 = to_jph(a->anchor_b);
        s.mAutoDetectPoint = false;
        return s.Create(b1, b2);
    });
}

bj_constraint bj_constraint_point(bj_world world, const bj_constraint_anchors *a) {
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        PointConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPoint1 = to_jph(a->anchor_a);
        s.mPoint2 = to_jph(a->anchor_b);
        return s.Create(b1, b2);
    });
}

bj_constraint bj_constraint_hinge(bj_world world, const bj_constraint_anchors *a,
                                   bj_vec3 axis, float limit_min, float limit_max)
{
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        HingeConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPoint1 = to_jph(a->anchor_a);
        s.mPoint2 = to_jph(a->anchor_b);
        Vec3 ax = to_jph(axis);
        s.mHingeAxis1 = ax;
        s.mHingeAxis2 = ax;
        Vec3 perp = ax.GetNormalizedPerpendicular();
        s.mNormalAxis1 = perp;
        s.mNormalAxis2 = perp;
        if (limit_min < limit_max) {
            s.mLimitsMin = limit_min;
            s.mLimitsMax = limit_max;
        }
        return s.Create(b1, b2);
    });
}

bj_constraint bj_constraint_slider(bj_world world, const bj_constraint_anchors *a,
                                    bj_vec3 axis, float limit_min, float limit_max)
{
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        SliderConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPoint1 = to_jph(a->anchor_a);
        s.mPoint2 = to_jph(a->anchor_b);
        Vec3 ax = to_jph(axis).Normalized();
        s.mSliderAxis1 = ax;
        s.mSliderAxis2 = ax;
        Vec3 perp = ax.GetNormalizedPerpendicular();
        s.mNormalAxis1 = perp;
        s.mNormalAxis2 = perp;
        if (limit_min < limit_max) {
            s.mLimitsMin = limit_min;
            s.mLimitsMax = limit_max;
        }
        return s.Create(b1, b2);
    });
}

bj_constraint bj_constraint_distance(bj_world world, const bj_constraint_anchors *a,
                                      float min_distance, float max_distance)
{
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        DistanceConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPoint1 = to_jph(a->anchor_a);
        s.mPoint2 = to_jph(a->anchor_b);
        s.mMinDistance = min_distance;
        s.mMaxDistance = max_distance;
        return s.Create(b1, b2);
    });
}

bj_constraint bj_constraint_six_dof(bj_world world, const bj_constraint_anchors *a,
                                     const float *trans_limits, const float *rot_limits)
{
    WorldImpl *w = as_world(world);
    if (!w || !a) return BJ_INVALID;
    return make_constraint(w, a, [&](Body &b1, Body &b2) -> TwoBodyConstraint * {
        SixDOFConstraintSettings s;
        s.mSpace = a->use_world_space ? EConstraintSpace::WorldSpace : EConstraintSpace::LocalToBodyCOM;
        s.mPosition1 = to_jph(a->anchor_a);
        s.mPosition2 = to_jph(a->anchor_b);
        s.mAxisX1 = s.mAxisX2 = Vec3(1, 0, 0);
        s.mAxisY1 = s.mAxisY2 = Vec3(0, 1, 0);
        if (trans_limits) {
            using EAxis = SixDOFConstraintSettings::EAxis;
            EAxis axes[3] = { EAxis::TranslationX, EAxis::TranslationY, EAxis::TranslationZ };
            for (int i = 0; i < 3; ++i) {
                float lo = trans_limits[i*2 + 0], hi = trans_limits[i*2 + 1];
                if (lo >= hi) { s.MakeFixedAxis(axes[i]); }
                else          { s.SetLimitedAxis(axes[i], lo, hi); }
            }
        }
        if (rot_limits) {
            using EAxis = SixDOFConstraintSettings::EAxis;
            EAxis axes[3] = { EAxis::RotationX, EAxis::RotationY, EAxis::RotationZ };
            for (int i = 0; i < 3; ++i) {
                float lo = rot_limits[i*2 + 0], hi = rot_limits[i*2 + 1];
                if (lo >= hi) { s.MakeFixedAxis(axes[i]); }
                else          { s.SetLimitedAxis(axes[i], lo, hi); }
            }
        }
        return s.Create(b1, b2);
    });
}

void bj_constraint_destroy(bj_world world, bj_constraint c) {
    WorldImpl *w = as_world(world);
    if (!w || c == BJ_INVALID) return;
    Constraint *cp = decode_constraint(c);
    w->system.RemoveConstraint(cp);
    cp->Release();   /* release our handle's ref */
}

void bj_constraint_set_enabled(bj_world world, bj_constraint c, uint8_t enabled) {
    (void)world;
    if (c == BJ_INVALID) return;
    decode_constraint(c)->SetEnabled(enabled != 0);
}

/* ================================================================== */
/*  Contact events                                                     */
/* ================================================================== */

uint32_t bj_world_contact_count(bj_world world) {
    WorldImpl *w = as_world(world);
    return w ? w->contacts.count() : 0;
}

uint32_t bj_world_pop_contacts(bj_world world, bj_contact *out, uint32_t max_out) {
    WorldImpl *w = as_world(world);
    return w ? w->contacts.drain(out, max_out) : 0;
}

void bj_world_clear_contacts(bj_world world) {
    if (auto *w = as_world(world)) w->contacts.clear();
}

/* ================================================================== */
/*  Character controller (CharacterVirtual)                            */
/* ================================================================== */
/* CharacterVirtual doesn't store its own layer so we wrap it in a     */
/* CharacterEntry that retains both the object + its configured layer. */

namespace {
struct CharacterEntry {
    Ref<CharacterVirtual> character;
    ObjectLayer           layer;
};
} /* namespace */

static inline CharacterEntry *decode_character(bj_character h) {
    return reinterpret_cast<CharacterEntry *>(h);
}

bj_character bj_character_create(bj_world world, bj_shape shape, const bj_character_desc *desc,
                                 bj_vec3 position, bj_quat rotation)
{
    WorldImpl *w = as_world(world);
    if (!w || !desc) return BJ_INVALID;
    const Shape *s = decode_shape(shape);
    if (!s) return BJ_INVALID;

    Ref<CharacterVirtualSettings> settings = new CharacterVirtualSettings();
    settings->mShape                      = s;
    settings->mMaxSlopeAngle              = desc->max_slope_angle;
    settings->mCharacterPadding           = desc->character_padding;
    settings->mPenetrationRecoverySpeed   = desc->penetration_recovery_speed;
    settings->mPredictiveContactDistance  = desc->predictive_contact_distance;
    settings->mMaxStrength                = desc->max_strength;
    settings->mMass                       = desc->mass;
    settings->mUp                         = to_jph(desc->up);
    /* Supporting-volume plane points up through feet; matches Jolt samples. */
    settings->mSupportingVolume           = Plane(to_jph(desc->up), -0.3f);

    CharacterVirtual *c = new CharacterVirtual(
        settings,
        to_jph(position), to_jph(rotation),
        /*user_data=*/0,
        &w->system
    );
    auto *entry = new CharacterEntry();
    entry->character = c;                     /* Ref<> AddRefs internally */
    entry->layer     = (ObjectLayer)desc->object_layer;
    return reinterpret_cast<bj_character>(entry);
}

void bj_character_destroy(bj_world /*world*/, bj_character h) {
    if (h == BJ_INVALID) return;
    delete decode_character(h);               /* Ref<> drops + deletes character */
}

void bj_character_update(bj_world world, bj_character h, float delta_time, bj_vec3 gravity) {
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    CharacterEntry *e = decode_character(h);

    /* Jolt's ExtendedUpdate uses `gravity` only for slope/direction decisions —
     * it does NOT integrate gravity into velocity. We do that here so the FFI
     * matches player intuition ("set move direction, update handles falling"). */
    Vec3 v = e->character->GetLinearVelocity();
    v += to_jph(gravity) * delta_time;
    e->character->SetLinearVelocity(v);

    CharacterVirtual::ExtendedUpdateSettings settings;   /* Jolt defaults */
    BodyFilter  body_filter;
    ShapeFilter shape_filter;
    e->character->ExtendedUpdate(
        delta_time, to_jph(gravity),
        settings,
        w->system.GetDefaultBroadPhaseLayerFilter(e->layer),
        w->system.GetDefaultLayerFilter(e->layer),
        body_filter, shape_filter,
        *w->temp_alloc
    );
}

void bj_character_get_position(bj_world /*world*/, bj_character h, bj_vec3 *out) {
    if (!out) return;
    if (h == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(decode_character(h)->character->GetPosition());
}
void bj_character_get_rotation(bj_world /*world*/, bj_character h, bj_quat *out) {
    if (!out) return;
    if (h == BJ_INVALID) { *out = { 0, 0, 0, 1 }; return; }
    *out = from_jph(decode_character(h)->character->GetRotation());
}
void bj_character_set_position(bj_world /*world*/, bj_character h, bj_vec3 p) {
    if (h == BJ_INVALID) return;
    decode_character(h)->character->SetPosition(to_jph(p));
}
void bj_character_set_rotation(bj_world /*world*/, bj_character h, bj_quat q) {
    if (h == BJ_INVALID) return;
    decode_character(h)->character->SetRotation(to_jph(q));
}

void bj_character_get_linear_velocity(bj_world /*world*/, bj_character h, bj_vec3 *out) {
    if (!out) return;
    if (h == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(decode_character(h)->character->GetLinearVelocity());
}
void bj_character_set_linear_velocity(bj_world /*world*/, bj_character h, bj_vec3 v) {
    if (h == BJ_INVALID) return;
    decode_character(h)->character->SetLinearVelocity(to_jph(v));
}

bj_ground_state bj_character_get_ground_state(bj_world /*world*/, bj_character h) {
    if (h == BJ_INVALID) return BJ_GROUND_IN_AIR;
    switch (decode_character(h)->character->GetGroundState()) {
        case CharacterBase::EGroundState::OnGround:      return BJ_GROUND_ON_GROUND;
        case CharacterBase::EGroundState::OnSteepGround: return BJ_GROUND_ON_STEEP;
        case CharacterBase::EGroundState::NotSupported:  return BJ_GROUND_NOT_SUPPORTED;
        case CharacterBase::EGroundState::InAir:         return BJ_GROUND_IN_AIR;
    }
    return BJ_GROUND_IN_AIR;
}
void bj_character_get_ground_normal(bj_world /*world*/, bj_character h, bj_vec3 *out) {
    if (!out) return;
    if (h == BJ_INVALID) { *out = { 0, 1, 0 }; return; }
    *out = from_jph(decode_character(h)->character->GetGroundNormal());
}
void bj_character_get_ground_position(bj_world /*world*/, bj_character h, bj_vec3 *out) {
    if (!out) return;
    if (h == BJ_INVALID) { *out = { 0, 0, 0 }; return; }
    *out = from_jph(decode_character(h)->character->GetGroundPosition());
}
bj_body bj_character_get_ground_body(bj_world /*world*/, bj_character h) {
    if (h == BJ_INVALID) return BJ_INVALID;
    return encode_body_id(decode_character(h)->character->GetGroundBodyID());
}

/* ================================================================== */
/*  Wheeled vehicles                                                   */
/* ================================================================== */
/* A VehicleEntry owns the chassis body + the VehicleConstraint + its  */
/* collision tester. Jolt's constraint stores a bare pointer to the    */
/* tester; it must outlive the constraint — that's why we wrap both.   */

namespace {
struct VehicleEntry {
    BodyID                         chassis_id;
    Ref<VehicleConstraint>         constraint;
    Ref<VehicleCollisionTesterRay> tester;
};
} /* namespace */

static inline VehicleEntry *decode_vehicle(bj_vehicle h) {
    return reinterpret_cast<VehicleEntry *>(h);
}

bj_vehicle bj_vehicle_create(bj_world world, bj_shape chassis_shape,
                             const bj_vehicle_desc *desc,
                             bj_vec3 position, bj_quat rotation)
{
    WorldImpl *w = as_world(world);
    if (!w || !desc) return BJ_INVALID;
    const Shape *shape = decode_shape(chassis_shape);
    if (!shape) return BJ_INVALID;

    BodyInterface &bi = w->system.GetBodyInterface();

    /* Create the chassis body — a regular dynamic body the vehicle constraint
     * drives via wheel forces. */
    BodyCreationSettings chassis_settings(
        shape,
        to_jph(position), to_jph(rotation),
        EMotionType::Dynamic, (ObjectLayer)desc->object_layer
    );
    chassis_settings.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    chassis_settings.mMassPropertiesOverride.mMass = 1500.0f;   /* typical car mass */
    BodyID chassis_id = bi.CreateAndAddBody(chassis_settings, EActivation::Activate);

    /* Vehicle constraint settings. */
    VehicleConstraintSettings vcs;
    vcs.mUp                  = to_jph(desc->up);
    vcs.mForward             = to_jph(desc->forward);
    vcs.mMaxPitchRollAngle   = desc->max_pitch_roll_angle;

    /* Four wheels — indices 0..3 matching bj_vehicle_desc::wheel_positions.
     * Front wheels steer; rear wheels drive. */
    for (int i = 0; i < 4; ++i) {
        auto *wheel = new WheelSettingsWV();
        wheel->mPosition             = to_jph(desc->wheel_positions[i]);
        wheel->mRadius               = desc->wheel_radius;
        wheel->mWidth                = desc->wheel_width;
        wheel->mSuspensionMinLength  = desc->suspension_min_length;
        wheel->mSuspensionMaxLength  = desc->suspension_max_length;
        wheel->mMaxSteerAngle        = (i < 2) ? desc->max_steer_angle : 0.0f;
        wheel->mMaxBrakeTorque       = desc->max_brake_torque;
        wheel->mMaxHandBrakeTorque   = (i >= 2) ? desc->max_handbrake_torque : 0.0f;
        vcs.mWheels.push_back(wheel);
    }

    /* Wheeled controller: one engine, automatic transmission defaults,
     * one differential driving the rear axle. */
    auto *controller = new WheeledVehicleControllerSettings();
    controller->mEngine.mMaxTorque = desc->engine_max_torque;
    VehicleDifferentialSettings diff;
    diff.mLeftWheel  = 2;   /* rear-left */
    diff.mRightWheel = 3;   /* rear-right */
    controller->mDifferentials.push_back(diff);
    vcs.mController = controller;

    /* Ray collision tester — one ray per wheel. Use non-moving layer for the
     * ground it collides against; for multi-layer setups, expose it later. */
    Ref<VehicleCollisionTesterRay> tester = new VehicleCollisionTesterRay(
        (ObjectLayer)BJ_LAYER_NON_MOVING,
        to_jph(desc->up)
    );

    /* Create the constraint against the (locked) chassis body. */
    VehicleConstraint *constraint = nullptr;
    {
        BodyLockWrite lock(w->system.GetBodyLockInterface(), chassis_id);
        if (!lock.Succeeded()) {
            bi.RemoveBody(chassis_id);
            bi.DestroyBody(chassis_id);
            return BJ_INVALID;
        }
        constraint = new VehicleConstraint(lock.GetBody(), vcs);
    }
    constraint->SetVehicleCollisionTester(tester);

    w->system.AddConstraint(constraint);
    /* Crucial: register as step-listener so the vehicle updates each physics step. */
    w->system.AddStepListener(constraint);

    auto *entry = new VehicleEntry();
    entry->chassis_id = chassis_id;
    entry->constraint = constraint;
    entry->tester     = tester;
    return reinterpret_cast<bj_vehicle>(entry);
}

void bj_vehicle_destroy(bj_world world, bj_vehicle h) {
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    VehicleEntry *e = decode_vehicle(h);
    w->system.RemoveStepListener(e->constraint);
    w->system.RemoveConstraint(e->constraint);
    BodyInterface &bi = w->system.GetBodyInterface();
    if (bi.IsAdded(e->chassis_id)) bi.RemoveBody(e->chassis_id);
    bi.DestroyBody(e->chassis_id);
    delete e;                   /* Ref<> drops constraint + tester */
}

bj_body bj_vehicle_get_chassis(bj_world /*world*/, bj_vehicle h) {
    if (h == BJ_INVALID) return BJ_INVALID;
    return encode_body_id(decode_vehicle(h)->chassis_id);
}

void bj_vehicle_set_input(bj_world world, bj_vehicle h,
                          float forward, float right, float brake, float handbrake)
{
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    VehicleEntry *e = decode_vehicle(h);
    auto *ctrl = static_cast<WheeledVehicleController *>(e->constraint->GetController());
    ctrl->SetDriverInput(forward, right, brake, handbrake);
    /* Wake the chassis body so throttle/brake take effect even after it's slept
     * on the suspension. Without this a car that came to rest ignores input. */
    if (forward != 0.0f || brake != 0.0f || handbrake != 0.0f || right != 0.0f) {
        w->system.GetBodyInterface().ActivateBody(e->chassis_id);
    }
}

float bj_vehicle_get_wheel_transform(bj_world world, bj_vehicle h,
                                     uint32_t wheel_index, uint32_t axis)
{
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID || wheel_index >= 4) return 0.0f;
    VehicleEntry *e = decode_vehicle(h);
    /* GetWheelWorldTransform needs wheel-local up + right axes. Jolt uses
     * the wheel's mWheelUp (default = chassis up) and mWheelForward for
     * orientation. Right-handed: we pick X as the wheel rotation axis. */
    Mat44 xform = e->constraint->GetWheelWorldTransform(
        wheel_index, Vec3::sAxisY(), Vec3::sAxisX()
    );
    if (axis < 3) {
        Vec3 pos = xform.GetTranslation();
        switch (axis) {
            case 0: return pos.GetX();
            case 1: return pos.GetY();
            default: return pos.GetZ();
        }
    } else if (axis < 7) {
        Quat rot = xform.GetQuaternion();
        switch (axis - 3) {
            case 0: return rot.GetX();
            case 1: return rot.GetY();
            case 2: return rot.GetZ();
            default: return rot.GetW();
        }
    }
    return 0.0f;
}

float bj_vehicle_get_engine_rpm(bj_world /*world*/, bj_vehicle h) {
    if (h == BJ_INVALID) return 0.0f;
    VehicleEntry *e = decode_vehicle(h);
    auto *ctrl = static_cast<WheeledVehicleController *>(e->constraint->GetController());
    return ctrl->GetEngine().GetCurrentRPM();
}

float bj_vehicle_get_wheel_angular_velocity(bj_world /*world*/, bj_vehicle h, uint32_t idx) {
    if (h == BJ_INVALID || idx >= 4) return 0.0f;
    VehicleEntry *e = decode_vehicle(h);
    const WheelWV *wheel = static_cast<const WheelWV *>(e->constraint->GetWheel(idx));
    return wheel ? wheel->GetAngularVelocity() : 0.0f;
}

/* ================================================================== */
/*  Soft bodies                                                        */
/* ================================================================== */

bj_body bj_soft_body_create(
    bj_world world,
    const float *vertex_data, uint32_t vertex_count,
    const uint32_t *indices,  uint32_t triangle_count,
    bj_vec3 position, bj_quat rotation,
    uint32_t object_layer,
    float edge_compliance, float gravity_factor, float linear_damping, float pressure)
{
    WorldImpl *w = as_world(world);
    if (!w || !vertex_data || !indices || vertex_count < 3 || triangle_count == 0) return BJ_INVALID;

    Ref<SoftBodySharedSettings> shared = new SoftBodySharedSettings();

    for (uint32_t i = 0; i < vertex_count; ++i) {
        SoftBodySharedSettings::Vertex v;
        v.mPosition = Float3(
            vertex_data[i * 4 + 0],
            vertex_data[i * 4 + 1],
            vertex_data[i * 4 + 2]
        );
        v.mVelocity = Float3(0.0f, 0.0f, 0.0f);
        v.mInvMass  = vertex_data[i * 4 + 3];
        shared->mVertices.push_back(v);
    }
    for (uint32_t i = 0; i < triangle_count; ++i) {
        SoftBodySharedSettings::Face f(indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2], 0);
        shared->AddFace(f);
    }
    /* Auto-generate edge + shear + bend constraints from the triangle topology. */
    SoftBodySharedSettings::VertexAttributes attrs;
    attrs.mCompliance         = edge_compliance;
    attrs.mShearCompliance    = edge_compliance;
    attrs.mBendCompliance     = edge_compliance;
    attrs.mLRAType            = SoftBodySharedSettings::ELRAType::None;
    shared->CreateConstraints(&attrs, 1, SoftBodySharedSettings::EBendType::Distance);
    shared->Optimize();

    SoftBodyCreationSettings bcs(
        shared.GetPtr(),
        to_jph(position), to_jph(rotation),
        (ObjectLayer)object_layer
    );
    bcs.mGravityFactor  = gravity_factor;
    bcs.mLinearDamping  = linear_damping;
    bcs.mPressure       = pressure;
    bcs.mUpdatePosition = true;

    BodyID id = w->system.GetBodyInterface().CreateAndAddSoftBody(bcs, EActivation::Activate);
    return encode_body_id(id);
}

static const SoftBodyMotionProperties *get_soft_mp(WorldImpl *w, bj_body h) {
    if (!w || h == BJ_INVALID) return nullptr;
    BodyLockRead lock(w->system.GetBodyLockInterface(), decode_body_id(h));
    if (!lock.Succeeded()) return nullptr;
    const Body &body = lock.GetBody();
    if (!body.IsSoftBody()) return nullptr;
    return static_cast<const SoftBodyMotionProperties *>(body.GetMotionPropertiesUnchecked());
}

uint32_t bj_soft_body_vertex_count(bj_world world, bj_body h) {
    WorldImpl *w = as_world(world);
    const SoftBodyMotionProperties *mp = get_soft_mp(w, h);
    return mp ? (uint32_t)mp->GetVertices().size() : 0;
}

void bj_soft_body_get_vertex(bj_world world, bj_body h, uint32_t idx, bj_vec3 *out) {
    if (!out) return;
    *out = { 0.0f, 0.0f, 0.0f };
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    BodyLockRead lock(w->system.GetBodyLockInterface(), decode_body_id(h));
    if (!lock.Succeeded()) return;
    const Body &body = lock.GetBody();
    if (!body.IsSoftBody()) return;
    const auto *mp = static_cast<const SoftBodyMotionProperties *>(body.GetMotionPropertiesUnchecked());
    if (!mp || idx >= mp->GetVertices().size()) return;
    /* SoftBody vertex positions are stored in body-local space; transform to world. */
    Vec3 local = mp->GetVertex(idx).mPosition;
    RMat44 xform = body.GetWorldTransform();
    Vec3 world_pos = Vec3(xform * RVec3(local));
    *out = from_jph(world_pos);
}

void bj_soft_body_set_vertex(bj_world world, bj_body h, uint32_t idx, bj_vec3 position) {
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(h));
    if (!lock.Succeeded()) return;
    Body &body = lock.GetBody();
    if (!body.IsSoftBody()) return;
    auto *mp = static_cast<SoftBodyMotionProperties *>(body.GetMotionPropertiesUnchecked());
    if (!mp || idx >= mp->GetVertices().size()) return;
    RMat44 xform = body.GetWorldTransform();
    RMat44 inv = xform.Inversed();
    Vec3 local = Vec3(inv * RVec3(to_jph(position)));
    mp->GetVertex(idx).mPosition = local;
}

void bj_soft_body_set_vertex_inv_mass(bj_world world, bj_body h, uint32_t idx, float inv_mass) {
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    BodyLockWrite lock(w->system.GetBodyLockInterface(), decode_body_id(h));
    if (!lock.Succeeded()) return;
    Body &body = lock.GetBody();
    if (!body.IsSoftBody()) return;
    auto *mp = static_cast<SoftBodyMotionProperties *>(body.GetMotionPropertiesUnchecked());
    if (!mp || idx >= mp->GetVertices().size()) return;
    mp->GetVertex(idx).mInvMass = inv_mass;
}

void bj_character_set_shape(bj_world world, bj_character h, bj_shape shape) {
    WorldImpl *w = as_world(world);
    if (!w || h == BJ_INVALID) return;
    const Shape *s = decode_shape(shape);
    if (!s) return;
    CharacterEntry *e = decode_character(h);
    BodyFilter body_filter;
    ShapeFilter shape_filter;
    e->character->SetShape(
        s,
        /*max_penetration_depth=*/FLT_MAX,
        w->system.GetDefaultBroadPhaseLayerFilter(e->layer),
        w->system.GetDefaultLayerFilter(e->layer),
        body_filter, shape_filter,
        *w->temp_alloc
    );
}

} /* extern "C" */
