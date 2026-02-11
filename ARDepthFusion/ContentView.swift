import SwiftUI
import SceneKit
import ARKit
import CoreImage
import CoreVideo

private let sharedCIContext = CIContext(options: [.useSoftwareRenderer: false])

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
    @State private var yoloLoaded = false
    @State private var depthLoaded = false
    @State private var effectsLoaded = false
    @State private var modelLoadError: String?

    private var modelsLoaded: Bool { yoloLoaded && depthLoaded && effectsLoaded }

    // Detection results screen state
    @State private var showDetectionResults = false
    @State private var capturedCIImage: CIImage?
    @State private var capturedIntrinsics: simd_float3x3?
    @State private var capturedCameraTransform: simd_float4x4?
    @State private var detectionElapsedMs: Double = 0

    // Per-detection depth samples (LiDAR-preferred, fused fallback)
    @State private var depthSamples: [DepthSample?] = []

    // Queued effects with pre-computed depth (placed after dismissing results screen)
    @State private var pendingEffects: [(type: EffectType, detection: DetectedObject, depth: DepthSample)] = []

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
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.5)
                            .tint(.white)
                        Text("Loading models...")
                            .foregroundStyle(.white)
                            .font(.headline)
                        VStack(alignment: .leading, spacing: 10) {
                            HStack(spacing: 8) {
                                Image(systemName: yoloLoaded ? "checkmark.circle.fill" : "circle.dashed")
                                    .foregroundStyle(yoloLoaded ? .green : .gray)
                                Text("YOLO Detection")
                            }
                            HStack(spacing: 8) {
                                Image(systemName: depthLoaded ? "checkmark.circle.fill" : "circle.dashed")
                                    .foregroundStyle(depthLoaded ? .green : .gray)
                                Text("Depth Anything")
                            }
                            HStack(spacing: 8) {
                                Image(systemName: effectsLoaded ? "checkmark.circle.fill" : "circle.dashed")
                                    .foregroundStyle(effectsLoaded ? .green : .gray)
                                Text("Video Effects")
                            }
                        }
                        .foregroundStyle(.white)
                        .font(.subheadline)
                    }
                }
            }
        }
        .task {
            let error = await ObjectDetectionService.shared.initialize()
            yoloLoaded = true
            if let error { modelLoadError = error }
        }
        .task {
            await DepthEstimator.preload()
            depthLoaded = true
        }
        .task {
            await effectManager.preloadVideos()
            await effectManager.prerollAllPlayers()
            effectsLoaded = true
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
                        // Look up the pre-computed depth sample for this detection
                        // so placement uses the same depth the user saw on screen.
                        let idx = detections.firstIndex(where: { $0.id == detection.id })
                        if let idx,
                           idx < depthSamples.count,
                           let sample = depthSamples[idx] {
                            pendingEffects.append((type: type, detection: detection, depth: sample))
                            effectManager.preparePlayer(for: type)
                        }
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

        let imgW = CVPixelBufferGetWidth(frame.capturedImage)
        let imgH = CVPixelBufferGetHeight(frame.capturedImage)
        imageWidth = CGFloat(imgW)
        imageHeight = CGFloat(imgH)

        capturedIntrinsics = frame.camera.intrinsics
        capturedCameraTransform = frame.camera.transform

        // Extract all data from ARFrame BEFORE the async Task.
        // This lets `frame` be released immediately, preventing ARSession starvation.
        // Copy pixel data: CIImage(cvPixelBuffer:) retains the buffer, starving ARSession.
        let cgCopy = sharedCIContext.createCGImage(
            CIImage(cvPixelBuffer: frame.capturedImage),
            from: CGRect(x: 0, y: 0, width: imgW, height: imgH)
        )
        if let cgCopy { capturedCIImage = CIImage(cgImage: cgCopy) }
        let bgraData = frame.capturedImage.toBGRAData()
        let lidarSnapshot = frame.sceneDepth.flatMap {
            LiDARSnapshot(depthMap: $0.depthMap, confidenceMap: $0.confidenceMap)
        }
        // frame is no longer needed — Swift releases it at end of scope

        Task {
            let start = CFAbsoluteTimeGetCurrent()

            async let yoloResult: [DetectedObject] = {
                if let bgra = bgraData {
                    return await ObjectDetectionService.shared.detect(
                        bgraData: bgra.data, width: bgra.width, height: bgra.height)
                }
                return []
            }()
            async let depthEstResult: DepthMapData? = {
                if let cgCopy {
                    return await DepthEstimator.shared?.estimateDepth(cgImage: cgCopy)
                }
                return nil
            }()

            let objects = await yoloResult
            let depthArray = await depthEstResult

            var fusionResult: DepthFusionResult?
            if let depthArray, let lidarSnapshot {
                fusionResult = DepthFusion.fuse(
                    relativeDepth: depthArray,
                    lidar: lidarSnapshot,
                    imageWidth: imgW,
                    imageHeight: imgH
                )
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Compute per-detection depth: prefer LiDAR, fall back to fused
            var samples: [DepthSample?] = []
            for obj in objects {
                let sample = sampleDepth(at: obj.centroid,
                                         lidar: lidarSnapshot,
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
        lidar: LiDARSnapshot?,
        fusionResult: DepthFusionResult?,
        imageWidth: Int,
        imageHeight: Int
    ) -> DepthSample? {
        if let lidar {
            let (lidarDepth, confidence) = sampleLiDAR(at: point, lidar: lidar,
                                                        imageWidth: imageWidth,
                                                        imageHeight: imageHeight)
            if let depth = lidarDepth,
               confidence >= 2,
               depth > 0.1 && depth < 5.0 && depth.isFinite {
                return DepthSample(depth: depth, isLiDAR: true)
            }
        }
        if let depth = fusionResult?.sampleDepth(at: point) {
            return DepthSample(depth: depth, isLiDAR: false)
        }
        return nil
    }

    /// Sample LiDAR depth at a camera-image-space point from a pre-copied snapshot.
    private func sampleLiDAR(
        at imagePoint: CGPoint,
        lidar: LiDARSnapshot,
        imageWidth: Int,
        imageHeight: Int
    ) -> (depth: Float?, confidence: UInt8) {
        let normX = Float(imagePoint.x) / Float(imageWidth)
        let normY = Float(imagePoint.y) / Float(imageHeight)
        let lx = min(Int(normX * Float(lidar.width)), lidar.width - 1)
        let ly = min(Int(normY * Float(lidar.height)), lidar.height - 1)

        guard lx >= 0, ly >= 0 else { return (nil, 0) }

        let idx = ly * lidar.width + lx
        return (lidar.depthValues[idx], lidar.confidenceValues[idx])
    }

    /// Place all queued effects after the detection results screen is dismissed.
    private func processPendingEffects() {
        let effects = pendingEffects
        let intrinsics = capturedIntrinsics
        let cameraTransform = capturedCameraTransform

        // Release heavy resources immediately
        pendingEffects.removeAll()
        capturedCIImage = nil
        capturedIntrinsics = nil
        capturedCameraTransform = nil
        depthResult = nil
        depthSamples = []

        guard !effects.isEmpty,
              let sceneView,
              let intrinsics,
              let cameraTransform else { return }

        // simd_float3x3 is column-major: matrix[column][row]
        let fx = intrinsics[0][0]  // column 0, row 0
        let fy = intrinsics[1][1]  // column 1, row 1
        let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
        let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

        for (type, detection, depthSample) in effects {
            // Use the pre-computed depth sample — same value the user saw on screen.
            let depth = depthSample.depth

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
                intrinsics: intrinsics
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
