import SwiftUI
import RealityKit
import ARKit
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
    @State private var selectedDetection: DetectedObject?
    @State private var showEffectPicker = false
    @State private var effectManager = EffectManager()

    var body: some View {
        ZStack {
            ARContainerView(coordinator: coordinator) { view in
                arView = view
            }
            .ignoresSafeArea()

            DetectionOverlayView(
                detections: detections,
                imageWidth: imageWidth,
                imageHeight: imageHeight,
                onTap: { detection in
                    selectedDetection = detection
                    showEffectPicker = true
                }
            )
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
        .sheet(isPresented: $showEffectPicker) {
            if let detection = selectedDetection {
                EffectPickerView(objectClass: detection.className) { effectType in
                    showEffectPicker = false
                    placeEffect(effectType, on: detection)
                }
                .presentationDetents([.height(280)])
            }
        }
    }

    private func runDetection() {
        guard !isDetecting else { return }
        guard let frame = coordinator.latestFrame else {
            statusText = "No AR frame available"
            return
        }

        isDetecting = true
        statusText = "Detecting..."
        detections = []
        imageWidth = CGFloat(CVPixelBufferGetWidth(frame.capturedImage))
        imageHeight = CGFloat(CVPixelBufferGetHeight(frame.capturedImage))

        Task {
            let start = CFAbsoluteTimeGetCurrent()

            async let yoloResult = ObjectDetectionService.shared.detect(frame: frame)
            async let depthEstResult = DepthEstimator.shared.estimateDepth(
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
            isDetecting = false

            if objects.isEmpty {
                statusText = "No objects detected"
            } else {
                statusText = String(
                    format: "%d objects (%.0fms)",
                    objects.count,
                    elapsed * 1000
                )
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

    private func placeEffect(_ type: ParticleEffectType, on detection: DetectedObject) {
        guard let arView, let frame = coordinator.latestFrame else { return }
        guard let fusion = depthResult else {
            statusText = "No depth data available"
            return
        }

        guard let depth = fusion.sampleDepth(at: detection.centroid) else {
            statusText = "No depth at object location"
            return
        }

        guard let worldPos = arView.worldPosition(
            imagePoint: detection.centroid,
            depth: depth,
            frame: frame
        ) else {
            statusText = "Could not compute 3D position"
            return
        }

        let scale = calculateEffectScale(
            boundingBox: detection.boundingBox,
            depth: depth,
            intrinsics: frame.camera.intrinsics,
            imageWidth: Float(CVPixelBufferGetWidth(frame.capturedImage))
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
