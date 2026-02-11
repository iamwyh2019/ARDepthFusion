# AR Depth Fusion - Real-time AR Effects App

## Project Overview

A real-time AR iOS app that detects objects and places 3D USDZ animation effects at their world positions. Uses LiDAR + Depth Anything fusion for accurate depth estimation.

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
| Output | USDZ 3D animations rendered in AR view |
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
| Confidence threshold | 0.7 |
| Effect types | Flamethrower, Explosion, Lightning, Dragon Breath, Smoke, Debug Cube (6 types) |
| UI Language | English |
| ML Compute Units | **CPU + Neural Engine only** (GPU reserved for SceneKit rendering) |

---

## Technical Stack

- **Language**: Swift 5.9+
- **UI**: SwiftUI
- **AR Framework**: SceneKit (`ARSCNView`) + USDZ animations
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
│   └── EffectType.swift                 # Effect type enum (6 types incl. debugCube)
│
├── Services/
│   ├── ObjectDetectionService.swift     # YOLO wrapper (C ABI bridge)
│   ├── DepthEstimator.swift             # Depth Anything CoreML inference
│   ├── DepthFusion.swift                # LiDAR + Depth Anything fusion
│   └── EffectManager.swift              # Manage placed effects (USDZ + debug cube)
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
- Confidence threshold: 0.7, IOU threshold: 0.5
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
class DepthFusion {

    struct Config {
        var minDepth: Float = 0.1          // meters
        var maxDepth: Float = 4.0          // meters
        var minConfidence: UInt8 = 2       // ARConfidenceLevel.high
    }

    struct FusionResult {
        let depthMap: [Float]              // Metric depth in meters
        let width: Int
        let height: Int
        let alpha: Float                   // Scale factor
        let beta: Float                    // Offset
    }

    var config = Config()

    /// Fuse relative depth with LiDAR absolute depth
    /// Model: D_metric = alpha * D_relative + beta
    func fuse(relativeDepth: MLMultiArray, arFrame: ARFrame) -> FusionResult? {

        guard let sceneDepth = arFrame.sceneDepth else { return nil }

        let lidarBuffer = sceneDepth.depthMap
        let confidenceBuffer = sceneDepth.confidenceMap

        let lidarWidth = CVPixelBufferGetWidth(lidarBuffer)
        let lidarHeight = CVPixelBufferGetHeight(lidarBuffer)
        let daWidth = relativeDepth.shape[2].intValue
        let daHeight = relativeDepth.shape[1].intValue

        // Collect valid pairs at LiDAR pixel locations
        var pairs: [(rel: Float, abs: Float)] = []

        CVPixelBufferLockBaseAddress(lidarBuffer, .readOnly)
        confidenceBuffer.map { CVPixelBufferLockBaseAddress($0, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(lidarBuffer, .readOnly)
            confidenceBuffer.map { CVPixelBufferUnlockBaseAddress($0, .readOnly) }
        }

        guard let depthPtr = CVPixelBufferGetBaseAddress(lidarBuffer)?
                .assumingMemoryBound(to: Float32.self) else { return nil }

        let confPtr = confidenceBuffer.flatMap {
            CVPixelBufferGetBaseAddress($0)?.assumingMemoryBound(to: UInt8.self)
        }

        for ly in 0..<lidarHeight {
            for lx in 0..<lidarWidth {
                let idx = ly * lidarWidth + lx
                let lidarDepth = depthPtr[idx]
                let confidence = confPtr?[idx] ?? 2

                // Filter by confidence and depth range
                guard confidence >= config.minConfidence,
                      lidarDepth > config.minDepth,
                      lidarDepth < config.maxDepth,
                      lidarDepth.isFinite else { continue }

                // Map LiDAR coord to Depth Anything coord (normalized)
                let normX = Float(lx) / Float(lidarWidth - 1)
                let normY = Float(ly) / Float(lidarHeight - 1)
                let daX = normX * Float(daWidth - 1)
                let daY = normY * Float(daHeight - 1)

                // Bilinear sample from Depth Anything
                let relDepth = bilinearSample(relativeDepth, x: daX, y: daY,
                                              width: daWidth, height: daHeight)

                guard relDepth.isFinite, relDepth > 0 else { continue }

                pairs.append((rel: relDepth, abs: lidarDepth))
            }
        }

        guard pairs.count >= 20 else { return nil }

        // Least squares fit: abs = alpha * rel + beta
        let (alpha, beta) = leastSquaresFit(pairs)

        guard alpha > 0.01, alpha < 100, beta.isFinite else { return nil }

        // Apply to full depth map
        var metricDepth = [Float](repeating: 0, count: daWidth * daHeight)
        for y in 0..<daHeight {
            for x in 0..<daWidth {
                let rel = relativeDepth[[0, y, x] as [NSNumber]].floatValue
                metricDepth[y * daWidth + x] = max(0.01, min(alpha * rel + beta, 100.0))
            }
        }

        return FusionResult(depthMap: metricDepth, width: daWidth, height: daHeight,
                           alpha: alpha, beta: beta)
    }

    private func bilinearSample(_ array: MLMultiArray, x: Float, y: Float,
                                width: Int, height: Int) -> Float {
        let x0 = Int(x), y0 = Int(y)
        let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
        let fx = x - Float(x0), fy = y - Float(y0)

        let v00 = array[[0, y0, x0] as [NSNumber]].floatValue
        let v10 = array[[0, y0, x1] as [NSNumber]].floatValue
        let v01 = array[[0, y1, x0] as [NSNumber]].floatValue
        let v11 = array[[0, y1, x1] as [NSNumber]].floatValue

        return v00*(1-fx)*(1-fy) + v10*fx*(1-fy) + v01*(1-fx)*fy + v11*fx*fy
    }

    private func leastSquaresFit(_ pairs: [(rel: Float, abs: Float)]) -> (Float, Float) {
        let n = Float(pairs.count)
        var sumX: Float = 0, sumY: Float = 0, sumXY: Float = 0, sumX2: Float = 0

        for p in pairs {
            sumX += p.rel; sumY += p.abs
            sumXY += p.rel * p.abs; sumX2 += p.rel * p.rel
        }

        let denom = n * sumX2 - sumX * sumX
        guard abs(denom) > 1e-10 else { return (1.0, 0.0) }

        let alpha = (n * sumXY - sumX * sumY) / denom
        let beta = (sumY * sumX2 - sumX * sumXY) / denom
        return (alpha, beta)
    }
}
```

### 4. SceneKit AR Setup

```swift
let sceneView = ARSCNView(frame: .zero)
sceneView.autoenablesDefaultLighting = true  // USDZ model illumination

let config = ARWorldTrackingConfiguration()
config.frameSemantics.insert(.sceneDepth)
config.planeDetection = [.horizontal, .vertical]
sceneView.session.run(config)
```

### 5. USDZ Effects (SceneKit)

6 effect types: flamethrower, explosion, lightning, dragonBreath, smoke, debugCube.

- USDZ files are loaded via `SCNScene(url:)`, children cloned and added to scene
- Animations are played recursively with `repeatCount = .infinity`
- Missing USDZ files are handled gracefully (print warning, no crash)
- `debugCube` places a 10cm red `SCNBox` (for position verification)
- Effects are `SCNNode`-based, removed via `removeFromParentNode()`

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

**ARFrame retention**: Do NOT store `ARFrame` — `CIImage(cvPixelBuffer:)` retains the CVPixelBuffer, starving ARSession. Instead, immediately copy pixels via `CIContext.createCGImage` → `CIImage(cgImage:)`, and store only `intrinsics` + `camera.transform`.

### 7. Effect Scale from Bounding Box

```swift
func calculateEffectScale(
    boundingBox: CGRect,      // Normalized 0-1
    depth: Float,             // Meters
    intrinsics: simd_float3x3,
    imageWidth: Float = 1920
) -> Float {
    let fx = intrinsics[0, 0]
    let pixelWidth = Float(boundingBox.width) * imageWidth
    let realWidth = (pixelWidth * depth) / fx
    return realWidth.clamped(to: 0.1...3.0)
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
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
│  Capture: copy pixel data (avoid ARFrame retention),             │
│           store intrinsics + camera.transform                    │
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
         │  1. Sample fused depth          │
         │  2. Unproject centroid → 3D     │
         │  3. Calculate scale from bbox   │
         │  4. EffectManager.placeEffect() │
         │     (USDZ load + animate)       │
         └─────────────────────────────────┘
                          │
                          ▼
         ┌─────────────────────────────────┐
         │  Back to Live AR View           │
         │  USDZ effects anchored in world │
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
- Confidence threshold: 0.7, IOU: 0.5
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
- USDZ loading via `SCNScene(url:)` (available since iOS 12)

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
- [ ] Debug cube placed at correct 3D world position
- [ ] USDZ effects load, animate, and position correctly (when files present)
- [ ] Missing USDZ files handled gracefully (warning, no crash)
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
