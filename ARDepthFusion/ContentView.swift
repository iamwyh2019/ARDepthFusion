import SwiftUI
import SceneKit
import ARKit
import CoreImage
import CoreVideo

struct ContentView: View {
    @StateObject private var coordinator = ARSessionCoordinator()
    @State private var sceneView: ARSCNView?
    @State private var detections: [DetectedObject] = []
    @State private var depthResult: DepthFusionResult?
    @State private var isDetecting = false
    @State private var statusText = "Ready"
    @State private var imageWidth: CGFloat = 1920
    @State private var imageHeight: CGFloat = 1440
    @StateObject private var effectManager = EffectManager()
    @State private var modelsLoaded = false
    @State private var modelLoadError: String?

    // Detection results screen state
    @State private var showDetectionResults = false
    @State private var capturedCIImage: CIImage?
    @State private var capturedIntrinsics: simd_float3x3?
    @State private var capturedCameraTransform: simd_float4x4?
    @State private var detectionElapsedMs: Double = 0

    // Per-detection depth samples (LiDAR-preferred, fused fallback)
    @State private var depthSamples: [DepthSample?] = []

    // Queued effects (placed after dismissing results screen)
    @State private var pendingEffects: [(type: EffectType, detection: DetectedObject)] = []

    var body: some View {
        ZStack {
            ARContainerView(coordinator: coordinator) { view in
                sceneView = view
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
                    EffectListView(effectManager: effectManager)
                }

                ControlPanelView(
                    statusText: statusText,
                    isDetecting: isDetecting,
                    detectionCount: detections.count,
                    effectCount: effectManager.placedEffects.count,
                    onDetect: { runDetection() },
                    onClearAll: {
                        effectManager.clearAll()
                        detections = []
                        depthResult = nil
                        depthSamples = []
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
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text("Loading models...")
                            .foregroundStyle(.white)
                            .font(.headline)
                    }
                }
            }
        }
        .task {
            async let yoloError = ObjectDetectionService.shared.initialize()
            async let depth: Void = DepthEstimator.preload()
            let error = await yoloError
            _ = await depth
            modelsLoaded = true
            if let error {
                modelLoadError = error
            }
        }
        .alert("Model Loading Error", isPresented: Binding(
            get: { modelLoadError != nil },
            set: { if !$0 { modelLoadError = nil } }
        )) {
            Button("OK") { modelLoadError = nil }
        } message: {
            Text(modelLoadError ?? "")
        }
        .fullScreenCover(isPresented: $showDetectionResults, onDismiss: processPendingEffects) {
            if let ciImage = capturedCIImage {
                DetectionResultsView(
                    capturedCIImage: ciImage,
                    detections: detections,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    elapsedMs: detectionElapsedMs,
                    depthSamples: depthSamples,
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

            // Compute per-detection depth: prefer LiDAR, fall back to fused
            let imgW = Int(imageWidth)
            let imgH = Int(imageHeight)
            var samples: [DepthSample?] = []
            for obj in objects {
                let sample = sampleDepth(at: obj.centroid, from: frame,
                                         fusionResult: fusionResult,
                                         imageWidth: imgW, imageHeight: imgH)
                samples.append(sample)
            }

            detections = objects
            depthResult = fusionResult
            depthSamples = samples
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
            }
            for (i, obj) in objects.enumerated() {
                let s = samples[i]
                let src = s?.isLiDAR == true ? "LiDAR" : "fused"
                print("  \(obj.className): depth=\(String(format: "%.2f", s?.depth ?? -1))m (\(src))")
            }
        }
    }

    /// Sample depth at a point: prefer high-confidence LiDAR, fall back to fused depth.
    private func sampleDepth(
        at point: CGPoint,
        from frame: ARFrame,
        fusionResult: DepthFusionResult?,
        imageWidth: Int,
        imageHeight: Int
    ) -> DepthSample? {
        let (lidarDepth, confidence) = sampleLiDAR(at: point, from: frame,
                                                    imageWidth: imageWidth,
                                                    imageHeight: imageHeight)
        if let depth = lidarDepth,
           confidence >= 2,
           depth > 0.1 && depth < 5.0 && depth.isFinite {
            return DepthSample(depth: depth, isLiDAR: true)
        }
        if let depth = fusionResult?.sampleDepth(at: point) {
            return DepthSample(depth: depth, isLiDAR: false)
        }
        return nil
    }

    /// Sample LiDAR depth map at a camera-image-space point.
    private func sampleLiDAR(
        at imagePoint: CGPoint,
        from frame: ARFrame,
        imageWidth: Int,
        imageHeight: Int
    ) -> (depth: Float?, confidence: UInt8) {
        guard let sceneDepth = frame.sceneDepth else { return (nil, 0) }

        let depthMap = sceneDepth.depthMap
        let lidarW = CVPixelBufferGetWidth(depthMap)
        let lidarH = CVPixelBufferGetHeight(depthMap)

        let normX = Float(imagePoint.x) / Float(imageWidth)
        let normY = Float(imagePoint.y) / Float(imageHeight)
        let lx = min(Int(normX * Float(lidarW)), lidarW - 1)
        let ly = min(Int(normY * Float(lidarH)), lidarH - 1)

        guard lx >= 0, ly >= 0 else { return (nil, 0) }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let depthPtr = CVPixelBufferGetBaseAddress(depthMap)?
                .assumingMemoryBound(to: Float32.self) else { return (nil, 0) }
        let depth = depthPtr[ly * lidarW + lx]

        var confidence: UInt8 = 0
        if let confMap = sceneDepth.confidenceMap {
            CVPixelBufferLockBaseAddress(confMap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confMap, .readOnly) }
            if let confPtr = CVPixelBufferGetBaseAddress(confMap)?
                    .assumingMemoryBound(to: UInt8.self) {
                confidence = confPtr[ly * lidarW + lx]
            }
        }

        return (depth, confidence)
    }

    /// Place all queued effects after the detection results screen is dismissed.
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
              let sceneView,
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
                in: sceneView
            )

            statusText = "\(type.displayName) placed on \(detection.className)"
        }
    }
}
