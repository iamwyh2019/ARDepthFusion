# AR Depth Fusion - Real-time AR Effects App

## Project Overview

A real-time AR iOS app that detects objects and places pre-rendered video effects at their world positions. Uses LiDAR + Depth Anything fusion for accurate depth estimation.

### Core Workflow
1. User opens app → loading screen while ML models preload → live AR camera view
2. User taps "Detect" button → captures still frame + camera pose
3. YOLO + Depth run in parallel → detection results screen appears (fullScreenCover)
4. Results screen: darkened background (30% brightness), detected objects highlighted at full brightness via segmentation masks, bounding boxes with centered labels showing class + distance
5. User taps a detected object → effect picker sheet → selects effect → effect queued
6. User taps "Back" → returns to live AR, queued effects placed at 3D world positions
7. User can add effects to multiple objects, delete individual ones

**This is a real-time AR app with a still-frame detection results screen for object selection.**

---

## Confirmed Requirements

| Requirement | Decision |
|-------------|----------|
| App Type | Real-time AR (live rendering) |
| Output | Pre-rendered video effects (.mov HEVC+alpha) on billboard planes in AR view |
| Orientation | Portrait only |
| Non-LiDAR devices | Not supported (App Store filters via `UIRequiredDeviceCapabilities`) |
| Effect size | Scaled based on bounding box size |
| Multi-object | Yes, different effects on different objects |
| UI style | Minimal (box only, tap to see class name) |
| Detection trigger | Manual (tap "Detect" button) |
| Effect anchoring | Fixed in world coordinates (doesn't follow object) |
| AR Framework | **SceneKit** with `ARSCNView` |
| Minimum iOS | **iOS 16.0** |
| Effect management | Can delete individual effects |
| Save feature | Not needed |
| Object detection | **Integrated YOLO source** (yolo11l-seg with segmentation masks) |
| Confidence threshold | 0.75 |
| Effect types | Explosion, Flamethrower, Smoke, Lightning, Magic, Snow, Tornado, Love, Aurora, Dance, Confetti, Debug Cube, Debug Mesh (13 types) |
| UI Language | English |
| ML Compute Units | **CPU + Neural Engine only** (GPU reserved for SceneKit rendering) |

---

## Technical Stack

- **Language**: Swift 5.9+
- **UI**: SwiftUI
- **AR Framework**: SceneKit (`ARSCNView`) + video effects (.mov HEVC with alpha)
- **Object Detection**: Integrated YOLO source (yolo11l-seg, with segmentation masks)
- **Depth Estimation**: CoreML (Depth Anything V2 Small F16)
- **Depth Source**: ARKit LiDAR (`ARFrame.sceneDepth`)
- **Minimum iOS**: 16.0
- **Required Hardware**: iPhone/iPad with LiDAR (iPhone 12 Pro+, iPad Pro 2020+)

---

## Project Structure

```
ARDepthFusion/
├── ARDepthFusionApp.swift               # App entry point
├── ContentView.swift                     # Main view, detection flow, effect placement
│
├── Models/
│   ├── DetectedObject.swift             # Detection result (bbox, class, mask, centroid)
│   ├── DepthFusionResult.swift          # Fused depth map model
│   ├── PlacedEffect.swift               # Placed effect tracking (SCNNode)
│   └── EffectType.swift                 # Effect type enum (13 types incl. debugCube/debugMesh)
│
├── Services/
│   ├── ObjectDetectionService.swift     # YOLO wrapper (C ABI bridge)
│   ├── DepthEstimator.swift             # Depth Anything CoreML inference
│   ├── DepthFusion.swift                # LiDAR + Depth Anything fusion
│   ├── EffectManager.swift              # Manage placed effects (video + debug cube/mesh)
│   └── VideoEffectNode.swift            # SCNNode subclass: AVPlayer + Metal texture billboard
│
├── Views/
│   ├── ARContainerView.swift            # SceneKit ARSCNView wrapper
│   ├── DetectionResultsView.swift       # Still-frame results with mask compositing + distance labels
│   ├── DetectionOverlayView.swift       # Live bounding box overlay (legacy)
│   ├── EffectPickerView.swift           # Effect selection sheet
│   ├── ControlPanelView.swift           # Detect button, etc.
│   └── EffectListView.swift             # List of placed effects (for deletion)
│
├── YOLO/
│   ├── YOLOBridge.swift                 # C ABI types (YOLODetection, callbacks)
│   ├── YOLOPredictor.swift              # CoreML inference + NMS + mask extraction
│   └── YOLOUtils.swift                  # Mask proto ops, sigmoid, crop, upsample
│
├── Utilities/
│   ├── CVPixelBuffer+Extensions.swift
│   ├── MLMultiArray+Extensions.swift
│   └── simd+Extensions.swift
│
├── Frameworks/
│   └── YOLOUnity.framework              # C ABI framework for YOLO model loading
│
├── DepthAnythingV2SmallF16.mlpackage    # Depth estimation model
└── Info.plist
```

---

## Key Implementation Details

### 1. CoreML Configuration (IMPORTANT)

**DO NOT use GPU for ML inference** - GPU is reserved for SceneKit rendering.

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine  // NOT .all, NOT .cpuAndGPU
```

### 2. YOLO Integration (yolo11l-seg)

YOLO is integrated via source code in `ARDepthFusion/YOLO/` with a C ABI bridge to `YOLOUnity.framework`:
- Model: `yolo11l-seg` (segmentation variant with instance masks)
- Confidence threshold: 0.75, IOU threshold: 0.5
- Returns per-detection: bounding box, class, confidence, centroid, segmentation mask
- Masks are smooth sigmoid values [0,1] at proto resolution (160x120), cropped to bbox
- Model files use hyphens (`yolo11l-seg`), but API expects underscores (`yolo11l_seg`)

### 2b. Detection Results Screen (DetectionResultsView)

Full-screen still-frame view shown after detection:
- **Compositing**: CIExposureAdjust (EV=-1.74, 30% brightness) for background, full brightness for masked regions via CIBlendWithMask
- **Mask upsampling**: vImage bilinear interpolation (`upsampleMask` in YOLOUtils) from proto 160x120 to full 1920x1440
- **Mask Y-flip**: Required because mask pixel data is Y=0-at-top but CIImage(cgImage:) composites with Y=0-at-bottom
- **Annotations**: Bounding boxes + two-line centered labels (class + distance) drawn directly on CGImage (pixel-perfect)
- **Distance labels**: Each detection shows `"className 95%\n2.30 m"` from depth fusion result
- **Landscape→Portrait rotation**: `.oriented(.right)` on CIImage; bbox mapping: `portrait_x = landscape_y, portrait_y = landscape_x`
- **Tap handling**: Converts view-space tap → portrait image space → find matching detection bbox
- **Effect flow**: Tap object → EffectPickerView sheet → effect queued → on dismiss, placed at 3D position

### 3. Depth Fusion Algorithm

Fuse Depth Anything (relative) with LiDAR (absolute) to get metric depth.

**Key principle**: Do NOT upsample LiDAR. Sample Depth Anything at LiDAR pixel locations.

```swift
/// LiDARSnapshot: pre-copied depth + confidence arrays from ARFrame.sceneDepth.
/// Created synchronously before async Task to avoid retaining ARFrame.
struct LiDARSnapshot {
    let depthValues: [Float]
    let confidenceValues: [UInt8]
    let width: Int
    let height: Int
    init?(depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer?)
}

enum DepthFusion {
    /// Fuse relative depth with LiDAR absolute depth
    /// Model: D_metric = alpha * D_relative + beta
    static func fuse(relativeDepth: DepthMapData, lidar: LiDARSnapshot,
                     imageWidth: Int, imageHeight: Int) -> DepthFusionResult?
    // Config: minConfidence >= 2, depth 0.1-4.0m, min 20 valid pairs
    // Iterates LiDAR pixels, samples Depth Anything via bilinear interp
}
```

### 4. SceneKit AR Setup

```swift
let sceneView = ARSCNView(frame: .zero)
sceneView.autoenablesDefaultLighting = true

let config = ARWorldTrackingConfiguration()
config.frameSemantics.insert(.sceneDepth)
config.planeDetection = [.horizontal, .vertical]
sceneView.session.run(config)
```

### 5. Video Effects (SceneKit)

13 effect types: explosion, flamethrower, smoke, lightning, magic, snow, tornado, love, aurora, dance, confetti, debugCube, debugMesh.

- Pre-rendered .mov videos (HEVC with alpha) rendered on `SCNPlane` billboards
- `AVPlayerItemVideoOutput` + `CVMetalTextureCache` pipeline for per-frame texture updates
- `Timer` (block API) scheduled via `DispatchQueue.main.async` for 60fps frame polling
- Players prerolled at launch (pool of 2 per type), auto-replenished on consumption
- Preroll requires `waitForReadyAndPreroll()` — KVO wait for `.readyToPlay` before calling `preroll(atRate:)`
- Looping via `AVPlayerItemDidPlayToEndTime` notification + seek-to-zero
- `SCNBillboardConstraint` with `freeAxes = [.Y]` (upright, faces camera horizontally)
- Shared `CVMetalTextureCache` across all nodes (owned by EffectManager, flushed on clearAll)
- `debugCube` places a red `SCNBox` sized and oriented to match the point-cloud OBB
- `debugMesh` places yellow 5mm `SCNSphere` dots at each mask-filtered, depth-filtered LiDAR world point (for verifying unprojection correctness)
- Effects are `SCNNode`-based, removed via `removeFromParentNode()` (triggers `stop()`)
- Depth for placement uses pre-computed `DepthSample` (same value user sees on detection screen)

### 5c. Point-Cloud OBB (Oriented Bounding Box)

The debug cube and 2D wireframe use a PCA-based OBB computed from mask-filtered LiDAR points:

1. **Depth filtering**: Mask-filtered LiDAR depths are sorted; only points within the 10th–90th percentile depth range are unprojected to 3D. This removes background leakage (desk, wall) that would otherwise elongate the point cloud along the camera's viewing direction and confuse PCA.

2. **PCA on XZ plane**: Compute 2×2 covariance matrix of world X,Z coordinates, find principal axis angle via `θ = atan2(2·Cxz, Cxx - Czz) / 2`. This aligns the box with the object's actual horizontal shape (e.g., a laptop's long axis).

3. **Min/max extents**: Rotate all filtered points into the PCA-aligned frame, compute min/max on each axis (rotated-X, Y, rotated-Z) with a 0.02m floor per dimension.

4. **Stored in `Object3DExtent`**: `obbCenter` (world), `obbDims` (width/height/depth in PCA frame), `obbYaw` (radians, angle from +X toward +Z), `obbPoints` (filtered 3D world points, used by debug mesh).

5. **Fallback**: If <10 mask-filtered 3D points after depth filtering, OBB fields are nil → falls back to 6-point camera-yaw-aligned method. `obbPoints` is still stored even with <10 points (debug mesh can still visualize sparse clouds).

**SceneKit yaw convention**: PCA yaw θ means the principal axis points at `(cos θ, 0, sin θ)` in world. `SCNNode.eulerAngles.y = -θ` aligns local X with this direction (SceneKit rotates local X toward −Z for positive angles).

**Consistency**: The wireframe (DetectionResultsView) and debug cube (EffectManager) use identical OBB data. The wireframe constructs axes directly from raw yaw; the debug cube uses negated yaw through SceneKit's euler convention. Both produce the same 8 world-space corners.

### 5b. VideoEffectNode Actor Isolation (CRITICAL)

**`VideoEffectNode` MUST be `nonisolated`** because SceneKit accesses `SCNNode` subclasses
from its internal rendering thread. With `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, omitting
`nonisolated` makes the class implicitly `@MainActor`-isolated, causing:
- Timer/CADisplayLink callbacks silently fail (`[weak self]` resolves to nil)
- `@MainActor @objc` methods incompatible with ObjC selector dispatch

```swift
@preconcurrency import SceneKit  // Suppress concurrency warnings for SceneKit types

nonisolated class VideoEffectNode: SCNNode, @unchecked Sendable {
    // Timer scheduled via DispatchQueue.main.async (not in init directly)
    // removeFromParentNode() override guarded with `if parent != nil`
    //   (SceneKit's addChildNode() internally calls removeFromParentNode())
}
```

### 6. 2D → 3D Unprojection

**CRITICAL**: `simd_float3x3` is **column-major** — `matrix[column][row]`.

```swift
// CORRECT:
let fx = intrinsics[0][0]  // column 0, row 0
let fy = intrinsics[1][1]  // column 1, row 1
let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

// WRONG (these are always 0):
let cx = intrinsics[0][2]  // column 0, row 2 — NOT the principal point!
let cy = intrinsics[1][2]  // column 1, row 2 — NOT the principal point!
```

Unprojection: pixel → camera space → world space:
```swift
let x = (Float(imagePoint.x) - cx) / fx * depth
let y = (Float(imagePoint.y) - cy) / fy * depth
let z = -depth  // Camera looks along -Z in ARKit
let cameraPoint = SIMD4<Float>(x, y, z, 1.0)
let worldPoint = cameraTransform * cameraPoint
```

**ARFrame retention**: Do NOT store `ARFrame` or pass it into async Tasks — `CIImage(cvPixelBuffer:)` retains the CVPixelBuffer, starving ARSession. Instead, extract everything synchronously before the Task:
- Pixels: `CIContext.createCGImage` → `CGImage` (used for both display and Depth Anything)
- YOLO: `toBGRAData()` → `Data` (pre-copied BGRA bytes)
- LiDAR: `LiDARSnapshot(depthMap:confidenceMap:)` → copied `[Float]` + `[UInt8]`
- Camera: store `intrinsics` + `camera.transform` as value types

### 7. Effect Scale from Bounding Box

```swift
func calculateEffectScale(
    boundingBox: CGRect,      // Pixel coordinates (not normalized)
    depth: Float,             // Meters
    intrinsics: simd_float3x3
) -> Float {
    let fx = intrinsics[0][0]
    let bboxWidthPixels = Float(boundingBox.width)
    let worldWidth = bboxWidthPixels * depth / fx
    return min(max(worldWidth, 0.1), 3.0)
}
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│              App Launch → Preload YOLO + DepthAnything            │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Live AR View (ARSCNView via ARContainerView)     │
└──────────────────────────────────────────────────────────────────┘
                               │
                    User taps "Detect" button
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Capture (sync, before async Task):                              │
│  - CGImage via CIContext.createCGImage (for display + depth)     │
│  - BGRA Data via toBGRAData() (for YOLO)                        │
│  - LiDARSnapshot (copied depth + confidence arrays)              │
│  - intrinsics + camera.transform (value types)                   │
│  ARFrame released immediately — no async retention               │
└──────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │ YOLO11l-seg │  │   Depth     │  │   LiDAR     │
      │ (det+masks) │  │  Anything   │  │   Depth     │
      │             │  │ CPU+NE only │  │  (raw)      │
      └─────────────┘  └─────────────┘  └─────────────┘
              │                │                │
              │                └───────┬────────┘
              │                        ▼
              │                ┌─────────────────┐
              │                │  DepthFusion    │
              │                │  α, β fitting   │
              │                └─────────────────┘
              │                        │
              ▼                        ▼
      ┌────────────────┐       ┌─────────────┐
      │ Detections     │       │Metric Depth │
      │ - bbox, class  │       │  (meters)   │
      │ - mask [0,1]   │       └─────────────┘
      │ - centroid     │              │
      └────────────────┘              │
              │                       │
              └───────────┬───────────┘
                          ▼
         ┌─────────────────────────────────┐
         │    DetectionResultsView         │
         │  (fullScreenCover)              │
         │                                 │
         │  Composited image:              │
         │  - Background at 30% brightness │
         │  - Masked objects full bright   │
         │  - Bboxes + centered labels     │
         │  - Distance from depth fusion   │
         │                                 │
         │  Tap object → EffectPicker      │
         │  → queue effect                 │
         └─────────────────────────────────┘
                          │
                    User taps "Back"
                          ▼
         ┌─────────────────────────────────┐
         │  processPendingEffects()        │
         │  For each queued effect:        │
         │  1. Use point-cloud OBB if      │
         │     available (PCA-aligned)     │
         │  2. Else: 6-point fallback      │
         │  3. Calculate scale from bbox   │
         │  4. EffectManager.placeEffect() │
         │     (video effect placement)    │
         └─────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────┐
         │  Back to Live AR View           │
         │  Video effects anchored in world│
         └─────────────────────────────────┘
```

---

## Info.plist

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSCameraUsageDescription</key>
    <string>Camera access is needed for AR experience</string>

    <key>UIRequiredDeviceCapabilities</key>
    <array>
        <string>arkit</string>
        <string>lidar</string>
    </array>

    <key>UISupportedInterfaceOrientations</key>
    <array>
        <string>UIInterfaceOrientationPortrait</string>
    </array>

    <key>UILaunchScreen</key>
    <dict/>
</dict>
</plist>
```

---

## Download Depth Anything Model

```bash
pip3 install huggingface_hub

huggingface-cli download \
  --local-dir ./models \
  apple/coreml-depth-anything-v2-small \
  --include "DepthAnythingV2SmallF16.mlpackage/*"
```

---

## UI Layout (Portrait)

### Main AR View
```
┌─────────────────────────────┐
│ LiDAR: ON         2 effects │
│                             │
│         AR Camera           │
│           View              │
│      (with placed effects)  │
│                             │
│  Effect List (swipe delete) │
│  [Detect]  Status  [Clear]  │
└─────────────────────────────┘
```

### Detection Results Screen (fullScreenCover)
```
┌─────────────────────────────┐
│ [< Back]                    │
│                             │
│  ┌───────────────────────┐  │
│  │ Darkened background   │  │
│  │                       │  │
│  │   ┌─────────┐         │  │  ← object at full brightness (mask)
│  │   │ cup 95% │         │  │  ← centered label + green bbox
│  │   │ 2.30 m  │         │  │  ← distance from depth fusion
│  │   └─────────┘         │  │
│  │                       │  │
│  └───────────────────────┘  │
│                             │
│  3 objects detected (245ms) │
└─────────────────────────────┘
```

---

## Critical Notes for Implementation

### ⚠️ YOLO Integration
- Model: `yolo11l-seg` (segmentation with instance masks)
- C ABI bridge via `YOLOUnity.framework` + Swift source in `YOLO/`
- Confidence threshold: 0.75, IOU: 0.5
- Masks: smooth sigmoid [0,1] at proto resolution (160x120), cropped to bbox
- Model file uses hyphens (`yolo11l-seg`), API expects underscores (`yolo11l_seg`)

### ⚠️ CoreML Compute Units
```swift
// CORRECT - GPU reserved for rendering
config.computeUnits = .cpuAndNeuralEngine

// WRONG - will compete with SceneKit
config.computeUnits = .all
```

### ⚠️ iOS 16 Minimum
- Uses `ObservableObject` + `@StateObject` (not `@Observable` which requires iOS 17)
- Uses `ARSCNView` (SceneKit, available since iOS 11)
- AVFoundation video playback (available since iOS 4)

### ⚠️ Coordinate Systems

| System | Origin | Y Direction | Notes |
|--------|--------|-------------|-------|
| Camera image | Top-left | Down | Landscape-right 1920x1440 |
| CIImage (CGImage-backed) | Bottom-left | Up | Y=0 at visual bottom |
| Portrait CGImage | Top-left | Down | After `.oriented(.right)`: 1440x1920 |
| UIKit / SwiftUI | Top-left | Down | Screen coordinates |

**Landscape→Portrait bbox mapping** (for CGImage-backed CIImage + `.oriented(.right)`):
```swift
// Landscape Y maps directly to portrait X (no flip needed)
func landscapeToPortrait(_ rect: CGRect) -> CGRect {
    CGRect(x: rect.minY, y: rect.minX, width: rect.height, height: rect.width)
}
```

**Mask Y-flip**: Mask pixel data is in pixel-buffer coords (Y=0 at top), but `CIImage(cgImage:)` composites with Y=0 at bottom. Must flip vertically when converting mask float→UInt8 for CGImage creation.

### ⚠️ Y-Flip Convention in `computeObject3DExtent` (HARD-WON LESSON)

Three separate Y-flips are required when mapping between YOLO bbox coords, LiDAR buffer coords, proto mask coords, and ARKit intrinsics. Missing any one causes subtle bugs (wrong depth sampling, lateral offset in 3D placement). These were discovered through iterative debugging with the debug mesh visualization.

| Step | From | To | Why flip? |
|------|------|----|-----------|
| **Bbox Y → LiDAR Y** | CIImage (Y=0 bottom) | Buffer (Y=0 top) | `toBGRAData()` renders through CIImage which flips Y. YOLO bbox Y is therefore CIImage convention. LiDAR buffer Y=0 is at scene top. |
| **LiDAR Y → Proto mask Y** | Buffer (Y=0 top) | CIImage (Y=0 bottom) | YOLO input was Y-flipped by CIContext.render, so the proto mask has Y=0 at bottom. Must flip LiDAR ly before sampling the mask. |
| **LiDAR Y → Image Y for unprojection** | Buffer (Y=0 top) | CIImage (Y=0 bottom) | ARKit intrinsics (cx, cy, fx, fy) project/unproject in CIImage convention. Using buffer-convention Y causes lateral shift in portrait (landscape Y → portrait X). |

```swift
// Bbox → LiDAR (flip Y):
let ly0 = max(0, Int((imgH - Float(bbox.maxY)) * scaleY))
let ly1 = min(lidar.height - 1, Int((imgH - Float(bbox.minY)) * scaleY))

// LiDAR → Proto mask (flip Y):
let protoY = Float(lidar.height - 1 - ly) * lidarToProtoY + protoPadY

// LiDAR → Image for unprojection (flip Y):
let imgY = imgH - Float(coord.ly) / scaleY
```

**Reference implementation**: `DetectionResultsView.buildDebugImage()` already had the first two flips correct (lines ~583-591, ~630). The unprojection flip matches `DetectionResultsView.unprojectToWorld()` which receives CIImage-convention coords.

**Diagnostic trick**: The debug mesh effect (yellow dots at LiDAR world points) makes coordinate bugs immediately visible — dots should overlap the detected object. Depth errors → dots lie flat on wrong surface. Lateral errors → dots shifted sideways.

### ⚠️ Depth Fusion
- **DO NOT upsample LiDAR**
- Sample Depth Anything at LiDAR pixel locations
- Use bilinear interpolation for sub-pixel accuracy

---

## Testing Checklist

- [ ] App launches with loading screen, models preload
- [ ] AR view displays camera feed
- [ ] `sceneDepth != nil` (LiDAR working)
- [ ] YOLO loads and detects with segmentation masks
- [ ] Depth Anything runs on CPU + Neural Engine
- [ ] Fusion produces valid alpha, beta
- [ ] Detection results screen appears with composited image
- [ ] Masks highlight objects at full brightness, background darkened
- [ ] Bounding boxes + labels align with objects
- [ ] Distance labels show correct depth values
- [ ] Tapping object shows effect picker
- [ ] Debug cube placed at correct 3D world position (PCA-aligned OBB)
- [ ] Debug cube wireframe on detection screen matches 3D debug cube orientation
- [ ] Debug cube aligns with object shape, not camera direction
- [ ] Debug mesh dots overlap the detected object (not shifted or on wrong surface)
- [ ] Video effects load, loop, and position correctly (when .mov files present)
- [ ] Missing .mov files handled gracefully (effect not shown in picker)
- [ ] Effect scale matches object size
- [ ] Multiple effects on different objects
- [ ] Delete individual effects works
- [ ] No ARFrame retention warnings (check console)
- [ ] Portrait lock enforced

---

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| YOLO (YOLOUnity) | < 100ms | Manual trigger only |
| Depth Anything | < 50ms | CPU + Neural Engine |
| Depth fusion | < 10ms | CPU only |
| AR rendering | 60 FPS | GPU |
