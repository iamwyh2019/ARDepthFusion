import SwiftUI
import RealityKit
import ARKit
import CoreImage
import CoreVideo

struct ContentView: View {
    @State private var coordinator = ARSessionCoordinator()
    @State private var arView: ARView?
    @State private var detections: [DetectedObject] = []
    @State private var depthResult: DepthFusionResult?
    @State private var isDetecting = false
    @State private var statusText = "Ready"
    @State private var imageWidth: CGFloat = 1920
    @State private var imageHeight: CGFloat = 1440
    @State private var effectManager = EffectManager()
    @State private var modelsLoaded = false

    // Detection results screen state
    @State private var showDetectionResults = false
    @State private var capturedCIImage: CIImage?
    @State private var capturedIntrinsics: simd_float3x3?
    @State private var capturedCameraTransform: simd_float4x4?
    @State private var detectionElapsedMs: Double = 0

    // Queued effects (placed after dismissing results screen)
    @State private var pendingEffects: [(type: ParticleEffectType, detection: DetectedObject)] = []

    var body: some View {
        ZStack {
            ARContainerView(coordinator: coordinator) { view in
                arView = view
            }
            .ignoresSafeArea()

            VStack {
                HStack {
                    Text(coordinator.hasLiDAR ? "LiDAR: ON" : "LiDAR: --")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                    Spacer()
                    if !effectManager.placedEffects.isEmpty {
                        Text("\(effectManager.placedEffects.count) effects")
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                }
                .padding(.horizontal)

                Spacer()

                if !effectManager.placedEffects.isEmpty {
                    EffectListView(effectManager: effectManager, arView: arView)
                }

                ControlPanelView(
                    statusText: statusText,
                    isDetecting: isDetecting,
                    detectionCount: detections.count,
                    effectCount: effectManager.placedEffects.count,
                    onDetect: { runDetection() },
                    onClearAll: {
                        if let arView {
                            effectManager.clearAll(from: arView)
                        }
                        detections = []
                        depthResult = nil
                        statusText = "Cleared"
                    }
                )
            }
        }
        .overlay {
            if !modelsLoaded {
                ZStack {
                    Color.black.ignoresSafeArea()
                    VStack(spacing: 16) {
                        ProgressView()
                            .controlSize(.large)
                            .tint(.white)
                        Text("Loading models...")
                            .foregroundStyle(.white)
                            .font(.headline)
                    }
                }
            }
        }
        .task {
            async let yolo: Void = ObjectDetectionService.shared.initialize()
            async let depth: Void = DepthEstimator.preload()
            _ = await (yolo, depth)
            modelsLoaded = true
        }
        .fullScreenCover(isPresented: $showDetectionResults, onDismiss: processPendingEffects) {
            if let ciImage = capturedCIImage {
                DetectionResultsView(
                    capturedCIImage: ciImage,
                    detections: detections,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    elapsedMs: detectionElapsedMs,
                    onPlaceEffect: { type, detection in
                        pendingEffects.append((type: type, detection: detection))
                    }
                )
            }
        }
    }

    private func runDetection() {
        guard modelsLoaded, !isDetecting else { return }
        guard let frame = coordinator.latestFrame else {
            statusText = "No AR frame available"
            return
        }

        isDetecting = true
        statusText = "Detecting..."
        detections = []
        imageWidth = CGFloat(CVPixelBufferGetWidth(frame.capturedImage))
        imageHeight = CGFloat(CVPixelBufferGetHeight(frame.capturedImage))

        // Copy pixel data immediately to avoid retaining the ARFrame's CVPixelBuffer.
        // CIImage(cvPixelBuffer:) holds a reference to the buffer, starving ARSession.
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])
        let tempCI = CIImage(cvPixelBuffer: frame.capturedImage)
        if let cgCopy = ciContext.createCGImage(tempCI, from: tempCI.extent) {
            capturedCIImage = CIImage(cgImage: cgCopy)
        }
        capturedIntrinsics = frame.camera.intrinsics
        capturedCameraTransform = frame.camera.transform

        Task {
            let start = CFAbsoluteTimeGetCurrent()

            async let yoloResult = ObjectDetectionService.shared.detect(frame: frame)
            async let depthEstResult = DepthEstimator.shared?.estimateDepth(
                pixelBuffer: frame.capturedImage
            )

            let objects = await yoloResult
            let depthArray = await depthEstResult

            var fusionResult: DepthFusionResult?
            if let depthArray {
                fusionResult = DepthFusion.fuse(
                    relativeDepth: depthArray,
                    arFrame: frame
                )
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            detections = objects
            depthResult = fusionResult
            detectionElapsedMs = elapsed * 1000
            isDetecting = false

            if objects.isEmpty {
                statusText = "No objects detected"
            } else {
                statusText = String(
                    format: "%d objects (%.0fms)",
                    objects.count,
                    elapsed * 1000
                )
                showDetectionResults = true
            }

            if let fusion = fusionResult {
                print("Fusion alpha=\(fusion.alpha) beta=\(fusion.beta) pairs=\(fusion.validPairCount)")
                for obj in objects {
                    let depth = fusion.sampleDepth(at: obj.centroid)
                    print("  \(obj.className): depth=\(String(format: "%.2f", depth ?? -1))m")
                }
            }
        }
    }

    /// Place all queued effects after the detection results screen is dismissed.
    /// Anchors are placed immediately; particle emitters are deferred by EffectManager
    /// to avoid Metal stencil errors while RealityKit's transparent render pass initializes.
    private func processPendingEffects() {
        let effects = pendingEffects
        let intrinsics = capturedIntrinsics
        let cameraTransform = capturedCameraTransform
        let fusion = depthResult
        let imgW = Float(imageWidth)

        // Release heavy resources immediately
        pendingEffects.removeAll()
        capturedCIImage = nil
        capturedIntrinsics = nil
        capturedCameraTransform = nil

        guard !effects.isEmpty,
              let arView,
              let intrinsics,
              let cameraTransform,
              let fusion else { return }

        // simd_float3x3 is column-major: matrix[column][row]
        let fx = intrinsics[0][0]  // column 0, row 0
        let fy = intrinsics[1][1]  // column 1, row 1
        let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
        let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

        for (type, detection) in effects {
            guard let depth = fusion.sampleDepth(at: detection.centroid) else { continue }

            // Inline world position computation (avoids needing ARFrame)
            let x = (Float(detection.centroid.x) - cx) / fx * depth
            let y = (Float(detection.centroid.y) - cy) / fy * depth
            let z = -depth  // Camera looks along -Z in ARKit
            let cameraPoint = SIMD4<Float>(x, y, z, 1.0)
            let worldPoint = cameraTransform * cameraPoint
            let worldPos = SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)

            let scale = calculateEffectScale(
                boundingBox: detection.boundingBox,
                depth: depth,
                intrinsics: intrinsics,
                imageWidth: imgW
            )

            effectManager.placeEffect(
                type: type,
                objectClass: detection.className,
                at: worldPos,
                scale: scale,
                in: arView
            )

            statusText = "\(type.displayName) placed on \(detection.className)"
        }
    }
}
