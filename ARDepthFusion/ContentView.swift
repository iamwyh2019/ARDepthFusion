import SwiftUI
import SceneKit
import ARKit
import CoreImage
import CoreVideo
import Accelerate

private let sharedCIContext = CIContext(options: [.useSoftwareRenderer: false])

/// Morphological erosion (min-filter) on a proto-resolution mask using vImage.
/// All-zero kernel = pure min-filter. kvImageEdgeExtend matches original boundary clamping.
nonisolated func erodeMask(_ mask: [Float], width: Int, height: Int, radius: Int) -> [Float] {
    guard radius > 0, mask.count == width * height else { return mask }
    let kernelSize = 2 * radius + 1
    let kernel = [Float](repeating: 0, count: kernelSize * kernelSize)
    var result = [Float](repeating: 0, count: width * height)
    mask.withUnsafeBufferPointer { srcPtr in
        result.withUnsafeMutableBufferPointer { dstPtr in
            var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                                    height: vImagePixelCount(height), width: vImagePixelCount(width),
                                    rowBytes: width * MemoryLayout<Float>.stride)
            var dst = vImage_Buffer(data: dstPtr.baseAddress!,
                                    height: vImagePixelCount(height), width: vImagePixelCount(width),
                                    rowBytes: width * MemoryLayout<Float>.stride)
            kernel.withUnsafeBufferPointer { kPtr in
                _ = vImageErode_PlanarF(&src, &dst, 0, 0, kPtr.baseAddress!,
                                        vImagePixelCount(kernelSize), vImagePixelCount(kernelSize),
                                        vImage_Flags(kvImageEdgeExtend))
            }
        }
    }
    return result
}

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

    // Per-detection 3D extents (LiDAR-preferred, fused fallback)
    @State private var objectExtents: [Object3DExtent?] = []

    // Queued effects with pre-computed extent (placed after dismissing results screen)
    @State private var pendingEffects: [(type: EffectType, detection: DetectedObject, extent: Object3DExtent)] = []

    // Detection task handle for cancellation
    @State private var detectionTask: Task<Void, Never>?

    // LiDAR snapshot for debug panel
    @State private var storedLidarSnapshot: LiDARSnapshot?

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
                        objectExtents = []
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
                    objectExtents: objectExtents,
                    intrinsics: capturedIntrinsics ?? simd_float3x3(1),
                    cameraTransform: capturedCameraTransform ?? simd_float4x4(1),
                    ciContext: sharedCIContext,
                    lidarSnapshot: storedLidarSnapshot,
                    onPlaceEffect: { type, detection in
                        let idx = detections.firstIndex(where: { $0.id == detection.id })
                        if let idx,
                           idx < objectExtents.count,
                           let extent = objectExtents[idx] {
                            pendingEffects.append((type: type, detection: detection, extent: extent))
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

        detectionTask?.cancel()
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
        storedLidarSnapshot = lidarSnapshot
        // frame is no longer needed — Swift releases it at end of scope

        detectionTask = Task {
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

            // Compute per-detection 3D extent: prefer LiDAR bbox sampling, fall back to fused
            let intrinsics = capturedIntrinsics ?? simd_float3x3(1)
            let camTransform = capturedCameraTransform
            var extents: [Object3DExtent?] = []
            for obj in objects {
                let extent = computeObject3DExtent(
                    detection: obj,
                    lidar: lidarSnapshot,
                    fusionResult: fusionResult,
                    imageWidth: imgW, imageHeight: imgH,
                    intrinsics: intrinsics,
                    cameraTransform: camTransform
                )
                extents.append(extent)
            }

            detections = objects
            depthResult = fusionResult
            objectExtents = extents
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

            #if DEBUG
            if let fusion = fusionResult {
                print("Fusion alpha=\(fusion.alpha) beta=\(fusion.beta) pairs=\(fusion.validPairCount)")
            }
            for (i, obj) in objects.enumerated() {
                let e = extents[i]
                let src = e?.isLiDAR == true ? "LiDAR" : "fused"
                print("  \(obj.className): depth=\(String(format: "%.2f", e?.medianDepth ?? -1))m (\(src))")
            }
            #endif
        }
    }

    /// Compute 3D extent for a detection by sampling LiDAR depths within its bbox.
    /// Falls back to fused depth if insufficient LiDAR coverage.
    private func computeObject3DExtent(
        detection: DetectedObject,
        lidar: LiDARSnapshot?,
        fusionResult: DepthFusionResult?,
        imageWidth: Int, imageHeight: Int,
        intrinsics: simd_float3x3,
        cameraTransform: simd_float4x4?
    ) -> Object3DExtent? {
        let bbox = detection.boundingBox
        // Bottom of object on screen = bbox.maxX in landscape, center = bbox.midY
        let bottomCenter = CGPoint(x: bbox.maxX, y: bbox.midY)

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
        let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

        // Try sampling LiDAR depths within the bbox
        if let lidar {
            let scaleX = Float(lidar.width) / Float(imageWidth)
            let scaleY = Float(lidar.height) / Float(imageHeight)

            let lx0 = max(0, Int(Float(bbox.minX) * scaleX))
            let ly0 = max(0, Int(Float(bbox.minY) * scaleY))
            let lx1 = min(lidar.width - 1, Int(Float(bbox.maxX) * scaleX))
            let ly1 = min(lidar.height - 1, Int(Float(bbox.maxY) * scaleY))

            // --- Mask-filtered depth sampling ---
            let mask = detection.mask
            let hasMask = mask.map({ !$0.isEmpty }) ?? false

            // Pre-compute LiDAR → proto mask coordinate mapping
            let protoW = detection.maskWidth
            let protoH = detection.maskHeight
            let modelToProto: Float = 4.0
            let modelW = Float(protoW) * modelToProto
            let modelH = Float(protoH) * modelToProto
            let s = min(modelW / Float(imageWidth), modelH / Float(imageHeight))
            let scaledW = Float(imageWidth) * s
            let scaledH = Float(imageHeight) * s
            let protoPadX = (modelW - scaledW) / 2.0 / modelToProto
            let protoPadY = (modelH - scaledH) / 2.0 / modelToProto
            let protoContentW = Float(protoW) - 2 * protoPadX
            let protoContentH = Float(protoH) - 2 * protoPadY
            let lidarToProtoX = protoContentW / Float(lidar.width)
            let lidarToProtoY = protoContentH / Float(lidar.height)

            // Erosion: 5% of shorter bbox dimension (in image pixels), converted to proto pixels
            // Clamped to [1, 3]: noise zone is ~1-3 proto pixels regardless of object size
            let erodedMask: [Float]?
            if hasMask {
                let erosionImagePx = 0.05 * min(Float(bbox.width), Float(bbox.height))
                let imageToProtoScale = protoContentW / Float(imageWidth)
                let erosionRadius = min(3, max(1, Int(round(erosionImagePx * imageToProtoScale))))
                erodedMask = erodeMask(mask!, width: protoW, height: protoH, radius: erosionRadius)
            } else {
                erodedMask = nil
            }

            // Bilinear sample mask at floating-point proto coordinates
            func sampleMask(_ m: [Float], px: Float, py: Float) -> Float {
                let x0 = max(0, Int(px))
                let y0 = max(0, Int(py))
                let x1 = min(x0 + 1, protoW - 1)
                let y1 = min(y0 + 1, protoH - 1)
                let fx = max(0, px - Float(x0))
                let fy = max(0, py - Float(y0))
                let v00 = m[y0 * protoW + x0]
                let v10 = m[y0 * protoW + x1]
                let v01 = m[y1 * protoW + x0]
                let v11 = m[y1 * protoW + x1]
                return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                     + v01 * (1 - fx) * fy + v11 * fx * fy
            }

            let bboxLidarPixels = max(0, (lx1 - lx0 + 1) * (ly1 - ly0 + 1))
            var maskedDepths: [Float] = []
            var allDepths: [Float] = []
            // Store LiDAR pixel coords for mask-filtered points (for deferred unprojection)
            var maskedLidarCoords: [(lx: Int, ly: Int, depth: Float)] = []
            maskedDepths.reserveCapacity(bboxLidarPixels)
            allDepths.reserveCapacity(bboxLidarPixels)
            maskedLidarCoords.reserveCapacity(bboxLidarPixels)

            if lx0 <= lx1 && ly0 <= ly1 {
            for ly in ly0...ly1 {
                for lx in lx0...lx1 {
                    let idx = ly * lidar.width + lx
                    let conf = lidar.confidenceValues[idx]
                    let d = lidar.depthValues[idx]
                    if conf >= 2 && d > 0.1 && d < 4.0 && d.isFinite {
                        allDepths.append(d)
                        if hasMask {
                            let protoX = Float(lx) * lidarToProtoX + protoPadX
                            let protoY = Float(ly) * lidarToProtoY + protoPadY
                            if sampleMask(erodedMask!, px: protoX, py: protoY) > 0.5 {
                                maskedDepths.append(d)
                                maskedLidarCoords.append((lx: lx, ly: ly, depth: d))
                            }
                        }
                    }
                }
            }
            }

            // Use mask-filtered depths if enough samples, otherwise fall back to bbox-only
            let useMask = maskedDepths.count >= 3
            var depths = useMask ? maskedDepths : allDepths
            #if DEBUG
            print("[DEBUG depth] \(detection.className): maskFiltered=\(maskedDepths.count)/\(allDepths.count) hasMask=\(hasMask) using=\(useMask ? "mask" : "bbox")")
            #endif

            if depths.count >= 3 {
                depths.sort()
                let medianDepth = depths[depths.count / 2]
                let depthMin = depths[depths.count / 20]
                let depthMax = depths[depths.count * 19 / 20]

                // Landscape width → portrait height, landscape height → portrait width
                let worldHeight = Float(bbox.width) * medianDepth / fx
                let worldWidth = Float(bbox.height) * medianDepth / fy

                // Unproject mask-filtered points within 10-90% depth range to 3D for OBB
                var pcObbCenter: SIMD3<Float>? = nil
                var pcObbDims: SIMD3<Float>? = nil
                var pcObbYaw: Float? = nil
                if let cam = cameraTransform, useMask {
                    let p10 = depths[depths.count * 5 / 100]
                    let p90 = depths[min(depths.count - 1, depths.count * 95 / 100)]
                    var maskedWorldPoints: [SIMD3<Float>] = []
                    maskedWorldPoints.reserveCapacity(maskedLidarCoords.count)
                    for coord in maskedLidarCoords {
                        if coord.depth >= p10 && coord.depth <= p90 {
                            let imgX = Float(coord.lx) / scaleX
                            let imgY = Float(coord.ly) / scaleY
                            let xCam = (imgX - cx) / fx * coord.depth
                            let yCam = (imgY - cy) / fy * coord.depth
                            let zCam = -coord.depth
                            let wp = cam * SIMD4<Float>(xCam, yCam, zCam, 1.0)
                            maskedWorldPoints.append(SIMD3(wp.x, wp.y, wp.z))
                        }
                    }

                    if maskedWorldPoints.count >= 10 {
                        let obb = computePointCloudOBB(maskedWorldPoints)
                        pcObbCenter = obb.center
                        pcObbDims = obb.dims
                        pcObbYaw = obb.yaw
                        #if DEBUG
                        print("[DEBUG OBB] \(detection.className): \(maskedWorldPoints.count)/\(maskedLidarCoords.count) pts (p10-p90), PCA yaw=\(String(format: "%.1f", obb.yaw * 180 / .pi))°, center=(\(String(format: "%.3f", obb.center.x)),\(String(format: "%.3f", obb.center.y)),\(String(format: "%.3f", obb.center.z))), dims=(\(String(format: "%.3f", obb.dims.x)),\(String(format: "%.3f", obb.dims.y)),\(String(format: "%.3f", obb.dims.z)))")
                        #endif
                    } else {
                        #if DEBUG
                        print("[DEBUG OBB] \(detection.className): \(maskedWorldPoints.count) points after depth filter, falling back to 6-point method")
                        #endif
                    }
                }

                return Object3DExtent(
                    bottomCenter: bottomCenter,
                    medianDepth: medianDepth,
                    depthMin: depthMin,
                    depthMax: depthMax,
                    worldWidth: worldWidth,
                    worldHeight: worldHeight,
                    isLiDAR: true,
                    obbCenter: pcObbCenter,
                    obbDims: pcObbDims,
                    obbYaw: pcObbYaw
                )
            }
        }

        // Fallback: fused depth at bottom center
        if let depth = fusionResult?.sampleDepth(at: bottomCenter) {
            let worldHeight = Float(bbox.width) * depth / fx
            let worldWidth = Float(bbox.height) * depth / fy

            return Object3DExtent(
                bottomCenter: bottomCenter,
                medianDepth: depth,
                depthMin: depth,
                depthMax: depth,
                worldWidth: worldWidth,
                worldHeight: worldHeight,
                isLiDAR: false,
                obbCenter: nil,
                obbDims: nil,
                obbYaw: nil
            )
        }

        return nil
    }

    /// Compute a gravity-aligned OBB from a 3D point cloud using PCA on the XZ plane.
    /// Points should be pre-filtered by depth (10-90%) before calling.
    /// Returns center, dimensions (width/height/depth in PCA-aligned frame), and yaw angle.
    private func computePointCloudOBB(_ points: [SIMD3<Float>]) -> (center: SIMD3<Float>, dims: SIMD3<Float>, yaw: Float) {
        // Step 1: PCA on XZ plane — single-pass Welford's algorithm for mean + covariance
        let n = Float(points.count)
        var meanX: Float = 0, meanZ: Float = 0
        var cxx: Float = 0, czz: Float = 0, cxz: Float = 0
        for (i, p) in points.enumerated() {
            let k = Float(i + 1)
            let dxOld = p.x - meanX
            let dzOld = p.z - meanZ
            meanX += dxOld / k
            meanZ += dzOld / k
            let dxNew = p.x - meanX
            let dzNew = p.z - meanZ
            cxx += dxOld * dxNew
            czz += dzOld * dzNew
            cxz += dxOld * dzNew
        }
        cxx /= n; czz /= n; cxz /= n

        // Angle of principal axis (eigenvector with larger eigenvalue)
        let yaw = atan2(2 * cxz, cxx - czz) / 2

        // Step 2: Rotate all points into PCA-aligned frame, find min/max on each axis
        let cosY = cos(-yaw)
        let sinY = sin(-yaw)

        var minRX: Float = .greatestFiniteMagnitude, maxRX: Float = -.greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude, maxY: Float = -.greatestFiniteMagnitude
        var minRZ: Float = .greatestFiniteMagnitude, maxRZ: Float = -.greatestFiniteMagnitude
        for p in points {
            let dx = p.x - meanX
            let dz = p.z - meanZ
            let rx = cosY * dx - sinY * dz
            let rz = sinY * dx + cosY * dz
            if rx < minRX { minRX = rx }; if rx > maxRX { maxRX = rx }
            if p.y < minY { minY = p.y }; if p.y > maxY { maxY = p.y }
            if rz < minRZ { minRZ = rz }; if rz > maxRZ { maxRZ = rz }
        }

        // Dims with minimum floor of 0.02m per axis
        let dimX = max(maxRX - minRX, 0.02)
        let dimY = max(maxY - minY, 0.02)
        let dimZ = max(maxRZ - minRZ, 0.02)

        // Center in PCA-aligned frame, then rotate back to world
        let midRX = (minRX + maxRX) / 2
        let midY = (minY + maxY) / 2
        let midRZ = (minRZ + maxRZ) / 2

        // Inverse rotation (yaw, not -yaw)
        let cosYInv = cos(yaw)
        let sinYInv = sin(yaw)
        let worldCenterX = cosYInv * midRX - sinYInv * midRZ + meanX
        let worldCenterZ = sinYInv * midRX + cosYInv * midRZ + meanZ

        let center = SIMD3<Float>(worldCenterX, midY, worldCenterZ)
        let dims = SIMD3<Float>(dimX, dimY, dimZ)

        return (center, dims, yaw)
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
        objectExtents = []
        storedLidarSnapshot = nil
        sharedCIContext.clearCaches()

        guard !effects.isEmpty,
              let sceneView,
              let intrinsics,
              let cameraTransform else { return }

        // simd_float3x3 is column-major: matrix[column][row]
        let fx = intrinsics[0][0]  // column 0, row 0
        let fy = intrinsics[1][1]  // column 1, row 1
        let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
        let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

        // Unproject helper: landscape image point at depth → world position
        func unproj(_ pt: CGPoint, d: Float) -> SIMD3<Float> {
            let px = (Float(pt.x) - cx) / fx * d
            let py = (Float(pt.y) - cy) / fy * d
            let cp = SIMD4<Float>(px, py, -d, 1.0)
            let wp = cameraTransform * cp
            return SIMD3(wp.x, wp.y, wp.z)
        }

        // Camera-yaw-aligned basis (gravity-aligned Y, camera-facing horizontal)
        let camZ = SIMD3<Float>(cameraTransform.columns.2.x, 0, cameraTransform.columns.2.z)
        let horizForward = normalize(SIMD3<Float>(-camZ.x, 0, -camZ.z))
        let up = SIMD3<Float>(0, 1, 0)
        let right = normalize(cross(horizForward, up))
        let cameraYaw = atan2(-camZ.x, -camZ.z)

        #if DEBUG
        print("[DEBUG placement] cameraYaw=\(String(format: "%.1f", cameraYaw * 180 / .pi))° camPos=(\(String(format: "%.3f", cameraTransform.columns.3.x)), \(String(format: "%.3f", cameraTransform.columns.3.y)), \(String(format: "%.3f", cameraTransform.columns.3.z)))")
        #endif

        for (type, detection, extent) in effects {
            let bbox = detection.boundingBox

            #if DEBUG
            print("[DEBUG placement] \(detection.className): bbox(landscape)=(\(String(format: "%.0f", bbox.minX)),\(String(format: "%.0f", bbox.minY)),\(String(format: "%.0f", bbox.width))x\(String(format: "%.0f", bbox.height)))")
            print("[DEBUG placement]   medianDepth=\(String(format: "%.3f", extent.medianDepth))m depthMin=\(String(format: "%.3f", extent.depthMin))m depthMax=\(String(format: "%.3f", extent.depthMax))m")
            #endif

            let effectPos: SIMD3<Float>
            let boxDimensions: SIMD3<Float>
            let effectYaw: Float

            if let obbC = extent.obbCenter, let obbD = extent.obbDims, let obbY = extent.obbYaw {
                // Point-cloud OBB path: use pre-computed OBB directly
                if type == .debugCube {
                    effectPos = obbC
                } else {
                    // Video effects: 2cm above bottom-center of OBB
                    let bottom = obbC.y - obbD.y / 2
                    let lift = min(0.02, obbD.y / 2)
                    effectPos = SIMD3<Float>(obbC.x, bottom + lift, obbC.z)
                }
                boxDimensions = obbD
                // Negate: PCA yaw is angle from +X toward +Z, but SCNNode.eulerAngles.y
                // rotates local X toward -Z. Negating aligns local X with the PCA axis.
                effectYaw = -obbY

                #if DEBUG
                print("[DEBUG OBB placement] \(detection.className): using point-cloud OBB, center=(\(String(format: "%.3f", obbC.x)),\(String(format: "%.3f", obbC.y)),\(String(format: "%.3f", obbC.z))), dims=(\(String(format: "%.3f", obbD.x)),\(String(format: "%.3f", obbD.y)),\(String(format: "%.3f", obbD.z))), yaw=\(String(format: "%.1f", obbY * 180 / .pi))°")
                #endif
            } else {
                // Fallback: 6-point unprojection method
                #if DEBUG
                print("[DEBUG OBB placement] \(detection.className): fallback to 6-point method")
                #endif
                let depth = extent.medianDepth
                let corners2D = [
                    CGPoint(x: bbox.minX, y: bbox.minY),
                    CGPoint(x: bbox.maxX, y: bbox.minY),
                    CGPoint(x: bbox.maxX, y: bbox.maxY),
                    CGPoint(x: bbox.minX, y: bbox.maxY),
                ]
                var worldPoints = corners2D.map { unproj($0, d: depth) }
                worldPoints.append(unproj(CGPoint(x: bbox.midX, y: bbox.midY), d: extent.depthMin))
                worldPoints.append(unproj(CGPoint(x: bbox.midX, y: bbox.midY), d: extent.depthMax))

                var minR: Float = .greatestFiniteMagnitude, maxR: Float = -.greatestFiniteMagnitude
                var minU: Float = .greatestFiniteMagnitude, maxU: Float = -.greatestFiniteMagnitude
                var minF: Float = .greatestFiniteMagnitude, maxF: Float = -.greatestFiniteMagnitude
                for (i, p) in worldPoints.enumerated() {
                    let r = dot(p, right); let u = dot(p, up); let f = dot(p, horizForward)
                    if i < 4 {
                        if r < minR { minR = r }; if r > maxR { maxR = r }
                        if u < minU { minU = u }; if u > maxU { maxU = u }
                    }
                    if f < minF { minF = f }; if f > maxF { maxF = f }
                }
                let midR = (minR + maxR) / 2
                let midU = (minU + maxU) / 2
                let midF = (minF + maxF) / 2
                let fallbackCenter = right * midR + up * midU + horizForward * midF
                let fallbackDims = SIMD3<Float>(maxR - minR, maxU - minU, maxF - minF)

                if type == .debugCube {
                    effectPos = fallbackCenter
                } else {
                    // Video effects: 2cm above bottom-center
                    let fallbackLift = min(0.02, (maxU - minU) / 2)
                    effectPos = right * midR + up * (minU + fallbackLift) + horizForward * midF
                }
                boxDimensions = fallbackDims
                effectYaw = cameraYaw
            }

            #if DEBUG
            print("[DEBUG placement]   effectPos=(\(String(format: "%.3f", effectPos.x)),\(String(format: "%.3f", effectPos.y)),\(String(format: "%.3f", effectPos.z)))")
            #endif

            let scale = calculateEffectScale(
                worldWidth: extent.worldWidth,
                worldHeight: extent.worldHeight
            )

            effectManager.placeEffect(
                type: type,
                objectClass: detection.className,
                at: effectPos,
                scale: scale,
                extent: extent,
                boxDimensions: boxDimensions,
                cameraYaw: effectYaw,
                in: sceneView
            )

            statusText = "\(type.displayName) placed on \(detection.className)"
        }
    }
}
