import SwiftUI
import SceneKit
import ARKit
import CoreImage
import CoreVideo
import Accelerate
import ImageIO

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
        let bgraData = frame.capturedImage.toPortraitBGRAData()
        // Debug: save the portrait image passed to YOLO
        if let bgra = bgraData {
            saveYOLOInputImage(bgraData: bgra.data, width: bgra.width, height: bgra.height)
        }
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

            // Convert portrait YOLO detections back to landscape coordinates.
            // Rotation was 90° CW (landscape→portrait), so inverse is 90° CCW.
            // Pixel mapping: portrait(px, py) → landscape(py, pW - 1 - px)
            // where pW = portrait image width = landscape image height.
            let pW = imgH  // portrait width = 1440
            let landscapeObjects: [DetectedObject] = objects.map { det in
                let pb = det.boundingBox
                let landscapeBbox = CGRect(
                    x: pb.minY,
                    y: CGFloat(pW - 1) - pb.maxX,
                    width: pb.height,
                    height: pb.width
                )
                let landscapeCentroid = CGPoint(
                    x: det.centroid.y,
                    y: CGFloat(pW - 1) - det.centroid.x
                )
                // Rotate proto mask 90° CCW: output[r][c] = input[c][N-1-r]
                // Proto is always N×N (160×160 for 640×640 model).
                var landscapeMask: [Float]? = nil
                if let mask = det.mask, !mask.isEmpty {
                    let N = det.maskWidth  // 160 (= maskHeight, always square)
                    var rotated = [Float](repeating: 0, count: N * N)
                    for r in 0..<N {
                        for c in 0..<N {
                            rotated[r * N + c] = mask[c * N + (N - 1 - r)]
                        }
                    }
                    landscapeMask = rotated
                }
                return DetectedObject(
                    className: det.className,
                    confidence: det.confidence,
                    boundingBox: landscapeBbox,
                    centroid: landscapeCentroid,
                    mask: landscapeMask,
                    maskWidth: det.maskWidth,
                    maskHeight: det.maskHeight
                )
            }

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
            for obj in landscapeObjects {
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

            detections = landscapeObjects
            depthResult = fusionResult
            objectExtents = extents
            detectionElapsedMs = elapsed * 1000
            isDetecting = false

            if landscapeObjects.isEmpty {
                statusText = "No objects detected"
            } else {
                statusText = String(
                    format: "%d objects (%.0fms)",
                    landscapeObjects.count,
                    elapsed * 1000
                )
                showDetectionResults = true
            }

            #if DEBUG
            if let fusion = fusionResult {
                print("Fusion alpha=\(fusion.alpha) beta=\(fusion.beta) pairs=\(fusion.validPairCount)")
            }
            for (i, obj) in landscapeObjects.enumerated() {
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

            // YOLO bbox Y is in CIImage convention (Y=0 at bottom) because
            // toBGRAData() renders through CIImage which flips Y. LiDAR depth Y=0
            // is at scene top. Flip bbox Y when mapping to LiDAR coords.
            let imgH = Float(imageHeight)
            let lx0 = max(0, Int(Float(bbox.minX) * scaleX))
            let ly0 = max(0, Int((imgH - Float(bbox.maxY)) * scaleY))
            let lx1 = min(lidar.width - 1, Int(Float(bbox.maxX) * scaleX))
            let ly1 = min(lidar.height - 1, Int((imgH - Float(bbox.minY)) * scaleY))

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
                            // Proto mask Y is flipped (YOLO input was Y-flipped by CIContext.render).
                            // LiDAR ly is Y=0-at-top; flip to match mask's Y=0-at-bottom convention.
                            let protoY = Float(lidar.height - 1 - ly) * lidarToProtoY + protoPadY
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
                var pcObbPoints: [SIMD3<Float>]? = nil
                if let cam = cameraTransform, useMask {
                    let p10 = depths[depths.count * 5 / 100]
                    let p90 = depths[min(depths.count - 1, depths.count * 95 / 100)]
                    var maskedWorldPoints: [SIMD3<Float>] = []
                    maskedWorldPoints.reserveCapacity(maskedLidarCoords.count)
                    for coord in maskedLidarCoords {
                        if coord.depth >= p10 && coord.depth <= p90 {
                            let imgX = Float(coord.lx) / scaleX
                            // LiDAR buffer Y=0 is at top, but ARKit intrinsics
                            // use CIImage convention (Y=0 at bottom). Flip Y.
                            let imgY = imgH - Float(coord.ly) / scaleY
                            let xCam = (imgX - cx) / fx * coord.depth
                            let yCam = (imgY - cy) / fy * coord.depth
                            let zCam = -coord.depth
                            let wp = cam * SIMD4<Float>(xCam, yCam, zCam, 1.0)
                            maskedWorldPoints.append(SIMD3(wp.x, wp.y, wp.z))
                        }
                    }

                    pcObbPoints = maskedWorldPoints

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
                    obbYaw: pcObbYaw,
                    obbPoints: pcObbPoints
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
                obbYaw: nil,
                obbPoints: nil
            )
        }

        return nil
    }

    /// Compute a gravity-aligned OBB from a 3D point cloud using a minimum-area bounding rectangle on the XZ plane.
    /// Points should be pre-filtered by depth (10-90%) before calling.
    /// Returns center, dimensions (width/height/depth in OBB-aligned frame), and yaw angle.
    private func computePointCloudOBB(_ points: [SIMD3<Float>]) -> (center: SIMD3<Float>, dims: SIMD3<Float>, yaw: Float) {
        // Step 0: Y extents (gravity axis, independent of XZ rotation) + XZ mean as rotation pivot
        var minY: Float = .greatestFiniteMagnitude, maxY: Float = -.greatestFiniteMagnitude
        var meanX: Float = 0, meanZ: Float = 0
        for p in points {
            if p.y < minY { minY = p.y }
            if p.y > maxY { maxY = p.y }
            meanX += p.x
            meanZ += p.z
        }
        let n = Float(points.count)
        meanX /= n; meanZ /= n

        // Step 1: Project to XZ, sort by X then Z (required by Andrew's monotone chain)
        var xz = points.map { ($0.x, $0.z) }
        xz.sort { $0.0 < $1.0 || ($0.0 == $1.0 && $0.1 < $1.1) }

        // Step 2: Convex hull
        let hull = convexHull(xz)

        // Step 3: Degenerate fallback — all points coincident in XZ
        if hull.count < 2 {
            let center = SIMD3<Float>(meanX, (minY + maxY) / 2, meanZ)
            return (center, SIMD3<Float>(0.02, max(maxY - minY, 0.02), 0.02), 0)
        }

        // Step 4: Minimum-area bounding rectangle — test each hull edge orientation
        var bestArea: Float = .greatestFiniteMagnitude
        var bestYaw: Float = 0
        var bestMinRX: Float = 0, bestMaxRX: Float = 0
        var bestMinRZ: Float = 0, bestMaxRZ: Float = 0

        let h = hull.count
        for i in 0..<h {
            let j = (i + 1) % h
            let dx = hull[j].0 - hull[i].0
            let dz = hull[j].1 - hull[i].1
            let edgeLen = dx * dx + dz * dz
            if edgeLen < 1e-12 { continue }

            let theta = atan2(dz, dx)
            let cosT = cos(-theta)
            let sinT = sin(-theta)

            // Rotate hull vertices only (not all points — hull is small)
            var eMinRX: Float = .greatestFiniteMagnitude, eMaxRX: Float = -.greatestFiniteMagnitude
            var eMinRZ: Float = .greatestFiniteMagnitude, eMaxRZ: Float = -.greatestFiniteMagnitude
            for v in hull {
                let cx = v.0 - meanX
                let cz = v.1 - meanZ
                let rx = cosT * cx - sinT * cz
                let rz = sinT * cx + cosT * cz
                if rx < eMinRX { eMinRX = rx }; if rx > eMaxRX { eMaxRX = rx }
                if rz < eMinRZ { eMinRZ = rz }; if rz > eMaxRZ { eMaxRZ = rz }
            }

            let area = (eMaxRX - eMinRX) * (eMaxRZ - eMinRZ)
            if area < bestArea {
                bestArea = area
                bestYaw = theta
                bestMinRX = eMinRX; bestMaxRX = eMaxRX
                bestMinRZ = eMinRZ; bestMaxRZ = eMaxRZ
            }
        }

        // Step 5: Normalize so dims.x >= dims.z (longer axis along yaw direction)
        var dimX = max(bestMaxRX - bestMinRX, 0.02)
        var dimZ = max(bestMaxRZ - bestMinRZ, 0.02)
        var midRX = (bestMinRX + bestMaxRX) / 2
        var midRZ = (bestMinRZ + bestMaxRZ) / 2

        if dimX < dimZ {
            bestYaw += .pi / 2
            swap(&dimX, &dimZ)
            let oldMidRX = midRX
            midRX = midRZ
            midRZ = -oldMidRX
        }

        let dimY = max(maxY - minY, 0.02)

        // Step 6: Reconstruct world center — inverse-rotate rotated-frame midpoint
        let cosYInv = cos(bestYaw)
        let sinYInv = sin(bestYaw)
        let worldCenterX = cosYInv * midRX - sinYInv * midRZ + meanX
        let worldCenterZ = sinYInv * midRX + cosYInv * midRZ + meanZ

        let center = SIMD3<Float>(worldCenterX, (minY + maxY) / 2, worldCenterZ)
        let dims = SIMD3<Float>(dimX, dimY, dimZ)

        return (center, dims, bestYaw)
    }

    /// 2D cross product: (a - o) × (b - o). Positive = left turn, zero = collinear, negative = right turn.
    private func cross2D(_ o: (Float, Float), _ a: (Float, Float), _ b: (Float, Float)) -> Float {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    }

    /// Andrew's monotone chain convex hull. Input must be sorted by X, then Y.
    /// Returns CCW-ordered hull vertices (no duplicated closing vertex).
    private func convexHull(_ sorted: [(Float, Float)]) -> [(Float, Float)] {
        let n = sorted.count
        if n < 2 { return sorted }

        var hull = [(Float, Float)]()
        hull.reserveCapacity(2 * n)

        // Lower hull (left → right)
        for p in sorted {
            while hull.count >= 2 && cross2D(hull[hull.count - 2], hull[hull.count - 1], p) <= 0 {
                hull.removeLast()
            }
            hull.append(p)
        }

        // Upper hull (right → left)
        let lowerCount = hull.count + 1
        for p in sorted.reversed() {
            while hull.count >= lowerCount && cross2D(hull[hull.count - 2], hull[hull.count - 1], p) <= 0 {
                hull.removeLast()
            }
            hull.append(p)
        }

        hull.removeLast() // Remove duplicated first point
        return hull
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
                worldPoints: extent.obbPoints,
                in: sceneView
            )

            statusText = "\(type.displayName) placed on \(detection.className)"
        }
    }
}

/// Save BGRA data as a PNG to Documents for debugging via the Files app.
/// The image is Y-flipped to show what YOLO actually sees (after bytesToCVPixelBuffer flipY).
private func saveYOLOInputImage(bgraData: Data, width: Int, height: Int) {
    let bytesPerRow = width * 4
    // Y-flip rows to show the image as YOLO actually sees it
    var flipped = Data(count: bgraData.count)
    bgraData.withUnsafeBytes { src in
        flipped.withUnsafeMutableBytes { dst in
            for y in 0..<height {
                memcpy(dst.baseAddress! + y * bytesPerRow,
                       src.baseAddress! + (height - 1 - y) * bytesPerRow,
                       bytesPerRow)
            }
        }
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue |
                                            CGBitmapInfo.byteOrder32Little.rawValue)

    guard let provider = CGDataProvider(data: flipped as CFData),
          let cgImage = CGImage(width: width, height: height,
                                bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: bytesPerRow,
                                space: colorSpace, bitmapInfo: bitmapInfo,
                                provider: provider, decode: nil,
                                shouldInterpolate: false, intent: .defaultIntent) else { return }

    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HHmmss"
    let filename = "yolo_input_\(formatter.string(from: Date())).png"
    let url = docs.appendingPathComponent(filename)

    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, cgImage, nil)
    CGImageDestinationFinalize(dest)
    print("Saved YOLO input: \(url.lastPathComponent) (\(width)×\(height))")
}
