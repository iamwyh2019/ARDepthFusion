# AR Depth Fusion - Real-time AR Effects App

## Project Overview

A real-time AR iOS app that detects objects and places 3D particle effects at their world positions. Uses LiDAR + Depth Anything fusion for accurate depth estimation and occlusion.

### Core Workflow
1. User opens app → live AR camera view
2. User taps "Detect" button → YOLO detects objects, shows bounding boxes
3. User taps a detected object → effect picker appears
4. User selects a particle effect (fire, smoke, etc.)
5. Effect is placed at the object's 3D world position (anchored, doesn't move)
6. User can add effects to multiple objects
7. User can delete individual effects

**This is a real-time AR app, NOT a photo capture + post-processing app.**

---

## Confirmed Requirements

| Requirement | Decision |
|-------------|----------|
| App Type | Real-time AR (live rendering) |
| Output | Particle effects rendered in AR view |
| Orientation | Portrait only |
| Non-LiDAR devices | Not supported (App Store filters via `UIRequiredDeviceCapabilities`) |
| Particle size | Scaled based on bounding box size |
| Occlusion | Depth-based occlusion (realistic) |
| Multi-object | Yes, different effects on different objects |
| UI style | Minimal (box only, tap to see class name) |
| Detection trigger | Manual (tap "Detect" button) |
| Effect anchoring | Fixed in world coordinates (doesn't follow object) |
| AR Framework | **RealityKit** with `ARView` |
| Minimum iOS | **iOS 18.0** (required for `ParticleEmitterComponent`) |
| Effect management | Can delete individual effects |
| Save feature | Not needed |
| Object detection | **YOLOUnity Framework** (user's custom plugin, uses YOLO11l) |
| Particle effects | Fire, Smoke, Sparks, Rain, Snow, Magic, Impact (all 7) |
| UI Language | English |
| ML Compute Units | **CPU + Neural Engine only** (GPU reserved for RealityKit rendering) |

---

## Technical Stack

- **Language**: Swift 5.9+
- **UI**: SwiftUI
- **AR Framework**: RealityKit (`ARView`, `ParticleEmitterComponent`)
- **Object Detection**: YOLOUnity.framework (external, provided by user)
- **Depth Estimation**: CoreML (Depth Anything V2 Small F16)
- **Depth Source**: ARKit LiDAR (`ARFrame.sceneDepth`)
- **Minimum iOS**: 18.0
- **Required Hardware**: iPhone/iPad with LiDAR (iPhone 12 Pro+, iPad Pro 2020+)

---

## Project Structure

```
ARDepthFusion/
├── App/
│   ├── ARDepthFusionApp.swift          # App entry point
│   └── ContentView.swift                # Main view container
│
├── Models/
│   ├── DetectedObject.swift             # Detection result model
│   ├── DepthFusionResult.swift          # Fused depth map model
│   ├── PlacedEffect.swift               # Placed effect tracking
│   └── ParticleEffectType.swift         # Effect type enum
│
├── Services/
│   ├── ObjectDetectionService.swift     # YOLOUnity wrapper
│   ├── DepthEstimator.swift             # Depth Anything CoreML inference
│   ├── DepthFusion.swift                # LiDAR + Depth Anything fusion
│   └── EffectManager.swift              # Manage placed effects
│
├── Views/
│   ├── ARContainerView.swift            # RealityKit ARView wrapper
│   ├── DetectionOverlayView.swift       # Bounding box overlay
│   ├── EffectPickerView.swift           # Effect selection sheet
│   ├── ControlPanelView.swift           # Detect button, etc.
│   └── EffectListView.swift             # List of placed effects (for deletion)
│
├── Utilities/
│   ├── CVPixelBuffer+Extensions.swift
│   ├── MLMultiArray+Extensions.swift
│   ├── simd+Extensions.swift
│   └── ARView+Extensions.swift
│
├── Frameworks/
│   └── YOLOUnity.framework              # User-provided YOLO framework
│
├── Resources/
│   ├── DepthAnythingV2SmallF16.mlpackage
│   └── Assets.xcassets
│
└── Info.plist
```

---

## Key Implementation Details

### 1. CoreML Configuration (IMPORTANT)

**DO NOT use GPU for ML inference** - GPU is reserved for RealityKit rendering.

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndNeuralEngine  // NOT .all, NOT .cpuAndGPU
```

### 2. YOLOUnity Framework Integration

The user has a custom `YOLOUnity.framework` that:
- Wraps YOLO11l model
- Handles post-processing (NMS, etc.)
- Returns detection results

**Claude Code should examine this framework to understand its API.**

Likely usage pattern (to be confirmed by examining the framework):
```swift
import YOLOUnity

// Hypothetical API - examine framework for actual interface
let detector = YOLODetector()
let results = detector.detect(pixelBuffer: frame.capturedImage)
// results likely contains: class, confidence, boundingBox, etc.
```

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

### 4. RealityKit AR Setup

```swift
import SwiftUI
import RealityKit
import ARKit

struct ARContainerView: UIViewRepresentable {
    @Binding var arView: ARView?
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        // Configure AR session with LiDAR depth
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        config.planeDetection = [.horizontal, .vertical]
        
        arView.session.run(config)
        
        // Enable depth-based occlusion
        arView.environment.sceneUnderstanding.options.insert(.occlusion)
        
        DispatchQueue.main.async {
            self.arView = arView
        }
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {}
}
```

### 5. Particle Effects (iOS 18+)

```swift
import RealityKit

enum ParticleEffectType: String, CaseIterable, Identifiable {
    case fire = "Fire"
    case smoke = "Smoke"
    case sparks = "Sparks"
    case rain = "Rain"
    case snow = "Snow"
    case magic = "Magic"
    case impact = "Impact"
    
    var id: String { rawValue }
    
    var icon: String {
        switch self {
        case .fire: return "🔥"
        case .smoke: return "💨"
        case .sparks: return "✨"
        case .rain: return "🌧️"
        case .snow: return "❄️"
        case .magic: return "🪄"
        case .impact: return "💥"
        }
    }
}

struct PlacedEffect: Identifiable {
    let id: UUID
    let type: ParticleEffectType
    let objectClass: String
    let anchor: AnchorEntity
}

@Observable
class EffectManager {
    var placedEffects: [PlacedEffect] = []
    
    func placeEffect(
        type: ParticleEffectType,
        objectClass: String,
        at worldPosition: SIMD3<Float>,
        scale: Float,
        in arView: ARView
    ) {
        let anchor = AnchorEntity(world: worldPosition)
        
        let entity = Entity()
        var emitter = ParticleEmitterComponent()
        configureEmitter(&emitter, for: type, scale: scale)
        entity.components.set(emitter)
        
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)
        
        let effect = PlacedEffect(
            id: UUID(),
            type: type,
            objectClass: objectClass,
            anchor: anchor
        )
        placedEffects.append(effect)
    }
    
    func removeEffect(_ effect: PlacedEffect) {
        effect.anchor.removeFromParent()
        placedEffects.removeAll { $0.id == effect.id }
    }
    
    func clearAll() {
        for effect in placedEffects {
            effect.anchor.removeFromParent()
        }
        placedEffects.removeAll()
    }
    
    private func configureEmitter(_ emitter: inout ParticleEmitterComponent,
                                  for type: ParticleEffectType, scale: Float) {
        switch type {
        case .fire:
            emitter.emitterShape = .cone
            emitter.emitterShapeSize = [0.1 * scale, 0.1 * scale, 0.1 * scale]
            emitter.birthRate = 200
            emitter.lifeSpan = 0.8
            emitter.speed = 0.3 * scale
            // Add color gradient: yellow → orange → red
            
        case .smoke:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = [0.15 * scale, 0.15 * scale, 0.15 * scale]
            emitter.birthRate = 80
            emitter.lifeSpan = 2.0
            emitter.speed = 0.15 * scale
            
        case .sparks:
            emitter.emitterShape = .point
            emitter.birthRate = 150
            emitter.lifeSpan = 0.5
            emitter.speed = 0.8 * scale
            
        case .rain:
            emitter.emitterShape = .plane
            emitter.emitterShapeSize = [0.5 * scale, 0, 0.5 * scale]
            emitter.birthRate = 300
            emitter.lifeSpan = 1.5
            emitter.speed = 2.0
            
        case .snow:
            emitter.emitterShape = .plane
            emitter.emitterShapeSize = [0.5 * scale, 0, 0.5 * scale]
            emitter.birthRate = 100
            emitter.lifeSpan = 3.0
            emitter.speed = 0.3
            
        case .magic:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = [0.2 * scale, 0.2 * scale, 0.2 * scale]
            emitter.birthRate = 50
            emitter.lifeSpan = 1.5
            emitter.speed = 0.2 * scale
            
        case .impact:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = [0.05 * scale, 0.05 * scale, 0.05 * scale]
            emitter.birthRate = 500
            emitter.lifeSpan = 0.4
            emitter.speed = 1.5 * scale
            emitter.birthRateVariation = 100
        }
    }
}
```

### 6. 2D → 3D Unprojection

```swift
extension ARView {
    
    /// Convert 2D screen point + depth to 3D world position
    func unproject(screenPoint: CGPoint, depth: Float) -> SIMD3<Float>? {
        guard let frame = session.currentFrame else { return nil }
        
        let intrinsics = frame.camera.intrinsics
        let imageRes = frame.camera.imageResolution
        
        // Screen → normalized → image coordinates
        let imageX = Float(screenPoint.x / bounds.width) * Float(imageRes.width)
        let imageY = Float(screenPoint.y / bounds.height) * Float(imageRes.height)
        
        let fx = intrinsics[0, 0]
        let fy = intrinsics[1, 1]
        let cx = intrinsics[2, 0]
        let cy = intrinsics[2, 1]
        
        // Unproject to camera space
        let z = depth
        let x = (imageX - cx) * z / fx
        let y = (imageY - cy) * z / fy
        
        // Transform to world space
        let cameraPoint = SIMD4<Float>(x, y, z, 1.0)
        let worldPoint = frame.camera.transform * cameraPoint
        
        return SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
    }
}
```

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
│                     App Running (Live AR View)                    │
└──────────────────────────────────────────────────────────────────┘
                               │
                    User taps "Detect" button
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  ARFrame                                                          │
│    ├── capturedImage (CVPixelBuffer, 1920×1440)                  │
│    ├── sceneDepth.depthMap (LiDAR, 256×192, Float32)             │
│    ├── sceneDepth.confidenceMap (256×192, UInt8)                 │
│    └── camera.intrinsics / camera.transform                       │
└──────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
      │ YOLOUnity   │  │   Depth     │  │   LiDAR     │
      │ Framework   │  │  Anything   │  │   Depth     │
      │ (YOLO11l)   │  │  (CoreML)   │  │  (raw)      │
      │             │  │ CPU+NE only │  │             │
      └─────────────┘  └─────────────┘  └─────────────┘
              │                │                │
              │                └───────┬────────┘
              │                        ▼
              │                ┌─────────────────┐
              │                │  DepthFusion    │
              │                │  α, β fitting   │
              │                │  D = α·rel + β  │
              │                └─────────────────┘
              │                        │
              ▼                        ▼
      ┌─────────────┐          ┌─────────────┐
      │ Detections  │          │Metric Depth │
      │ - bbox      │          │  (meters)   │
      │ - class     │          │  518×518    │
      │ - score     │          └─────────────┘
      └─────────────┘                 │
              │                       │
              └───────────┬───────────┘
                          ▼
              ┌─────────────────────┐
              │ Show bounding boxes │
              │ (tap to see class)  │
              └─────────────────────┘
                          │
                  User taps a box
                          ▼
              ┌─────────────────────┐
              │  EffectPickerView   │
              │ 🔥💨✨🌧️❄️🪄💥       │
              └─────────────────────┘
                          │
                  User selects effect
                          ▼
              ┌─────────────────────┐
              │ 1. Get bbox center  │
              │ 2. Sample depth     │
              │ 3. Unproject → 3D   │
              │ 4. Calc scale       │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  EffectManager      │
              │  .placeEffect()     │
              │  → AnchorEntity     │
              │  → Particle emitter │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Effect renders with │
              │ depth-based         │
              │ occlusion           │
              └─────────────────────┘
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

```
┌─────────────────────────────┐
│         Status Bar          │
├─────────────────────────────┤
│                             │
│                             │
│         AR Camera           │
│           View              │
│                             │
│      ┌─────────┐            │
│      │ object  │ ← tap to   │
│      └─────────┘   add fx   │
│                             │
│                             │
├─────────────────────────────┤
│                             │
│  [ 🔍 Detect ]  [ 🗑 Clear ] │
│                             │
│  Active Effects:            │
│  🔥 cup  ×    💨 chair  ×   │ ← tap × to delete
│                             │
└─────────────────────────────┘
```

---

## Critical Notes for Implementation

### ⚠️ YOLOUnity Framework
- User-provided `.framework` file
- **Examine headers/API before implementing**
- Uses YOLO11l model internally
- Handles NMS post-processing

### ⚠️ CoreML Compute Units
```swift
// CORRECT - GPU reserved for rendering
config.computeUnits = .cpuAndNeuralEngine

// WRONG - will compete with RealityKit
config.computeUnits = .all
```

### ⚠️ iOS 18 Requirement
- `ParticleEmitterComponent` requires iOS 18+
- Set deployment target to iOS 18.0

### ⚠️ Coordinate Systems
| System | Origin | Y Direction |
|--------|--------|-------------|
| Vision | Bottom-left | Up |
| UIKit | Top-left | Down |
| RealityKit | Center | Up |

Convert bounding boxes from Vision to screen coordinates:
```swift
// Vision bbox (bottom-left origin, normalized)
let visionRect = detection.boundingBox

// Convert to UIKit (top-left origin)
let screenRect = CGRect(
    x: visionRect.minX * viewWidth,
    y: (1 - visionRect.maxY) * viewHeight,  // Flip Y
    width: visionRect.width * viewWidth,
    height: visionRect.height * viewHeight
)
```

### ⚠️ Depth Fusion
- **DO NOT upsample LiDAR**
- Sample Depth Anything at LiDAR pixel locations
- Use bilinear interpolation for sub-pixel accuracy

---

## Testing Checklist

- [ ] AR view displays camera feed
- [ ] `sceneDepth != nil` (LiDAR working)
- [ ] YOLOUnity loads and detects
- [ ] Depth Anything runs on CPU + Neural Engine
- [ ] Fusion produces valid α (0.5-5.0 typical), β
- [ ] Bounding boxes appear on tap "Detect"
- [ ] Tapping box shows effect picker
- [ ] Effects appear at correct 3D position
- [ ] Effect scale matches object size
- [ ] Occlusion works (fx hidden behind objects)
- [ ] Multiple effects on different objects
- [ ] Delete individual effects works
- [ ] Portrait lock enforced

---

## Performance Targets

| Component | Target | Notes |
|-----------|--------|-------|
| YOLO (YOLOUnity) | < 100ms | Manual trigger only |
| Depth Anything | < 50ms | CPU + Neural Engine |
| Depth fusion | < 10ms | CPU only |
| AR rendering | 60 FPS | GPU |
