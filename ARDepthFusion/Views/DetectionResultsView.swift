import SwiftUI
import CoreImage
import CoreImage.CIFilterBuiltins
import Accelerate
import simd

struct DetectionResultsView: View {
    let capturedCIImage: CIImage
    let detections: [DetectedObject]
    let imageWidth: CGFloat
    let imageHeight: CGFloat
    let elapsedMs: Double
    let objectExtents: [Object3DExtent?]
    let intrinsics: simd_float3x3
    let cameraTransform: simd_float4x4
    let ciContext: CIContext
    let lidarSnapshot: LiDARSnapshot?
    let onPlaceEffect: (EffectType, DetectedObject) -> Void

    private enum ViewMode: String, CaseIterable {
        case effect = "Effect"
        case debug = "Debug"
    }

    @Environment(\.dismiss) private var dismiss
    @State private var composited: UIImage?
    @State private var selectedDetection: DetectedObject?
    @State private var statusText = ""
    @State private var viewMode: ViewMode = .effect
    @State private var debugImage: UIImage?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            switch viewMode {
            case .effect:
                if let composited {
                    GeometryReader { geometry in
                        Image(uiImage: composited)
                            .resizable()
                            .scaledToFill()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .clipped()
                            .contentShape(Rectangle())
                            .onTapGesture { location in
                                handleTap(at: location, viewSize: geometry.size,
                                          imageSize: composited.size)
                            }
                    }
                    .ignoresSafeArea()
                }
            case .debug:
                if let debugImage {
                    GeometryReader { geometry in
                        Image(uiImage: debugImage)
                            .resizable()
                            .scaledToFill()
                            .frame(width: geometry.size.width, height: geometry.size.height)
                            .clipped()
                    }
                    .ignoresSafeArea()
                } else if lidarSnapshot != nil {
                    ProgressView("Rendering debug view...")
                        .foregroundStyle(.white)
                } else {
                    Text("No LiDAR data available")
                        .foregroundStyle(.secondary)
                }
            }

            VStack {
                HStack {
                    Button { dismiss() } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "chevron.left")
                            Text("Back")
                        }
                        .font(.body.weight(.medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                    }
                    Spacer()
                    if lidarSnapshot != nil {
                        Picker("View", selection: $viewMode) {
                            ForEach(ViewMode.allCases, id: \.self) { mode in
                                Text(mode.rawValue).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 160)
                    }
                }
                .padding(.horizontal)

                Spacer()

                if !statusText.isEmpty {
                    Text(statusText)
                        .font(.caption)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }

                if viewMode == .debug {
                    Text("Green = mask edge, Red = eroded edge")
                        .font(.caption2)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }

                Text("\(detections.count) objects detected (\(Int(elapsedMs))ms)")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.bottom, 8)
            }
        }
        .task {
            composited = buildCompositedImage()
        }
        .onChange(of: viewMode) {
            if viewMode == .debug && debugImage == nil {
                debugImage = buildDebugImage()
            }
        }
        .sheet(item: $selectedDetection) { detection in
            EffectPickerView(objectClass: detection.className) { effectType in
                selectedDetection = nil
                onPlaceEffect(effectType, detection)
                statusText = "\(effectType.displayName) queued for \(detection.className)"
            }
            .presentationDetents([.height(280)])
        }
    }

    // MARK: - Tap Handling

    /// Convert tap in view space → portrait image space → find matching detection
    private func handleTap(at location: CGPoint, viewSize: CGSize, imageSize: CGSize) {
        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let offsetX = (imageSize.width * scale - viewSize.width) / 2
        let offsetY = (imageSize.height * scale - viewSize.height) / 2

        let imgX = (location.x + offsetX) / scale
        let imgY = (location.y + offsetY) / scale
        let tapPoint = CGPoint(x: imgX, y: imgY)

        for detection in detections {
            let portRect = landscapeToPortrait(detection.boundingBox)
            if portRect.contains(tapPoint) {
                selectedDetection = detection
                return
            }
        }
    }

    /// Landscape image bbox → portrait image bbox.
    /// The CIImage (CGImage-backed) composites with Y=0 at bottom, so after
    /// .oriented(.right), landscape Y maps directly to portrait X (no flip).
    private func landscapeToPortrait(_ rect: CGRect) -> CGRect {
        CGRect(
            x: rect.minY,
            y: rect.minX,
            width: rect.height,
            height: rect.width
        )
    }

    // MARK: - Image Compositing

    private func buildCompositedImage() -> UIImage? {
        let original = capturedCIImage

        // CIExposureAdjust preserves hue. EV = log2(0.3) ≈ -1.74 → 0.3x brightness
        let darkened = original.applyingFilter("CIExposureAdjust", parameters: [
            kCIInputEVKey: -1.74
        ])

        // Brighten masked objects by 1.2x for greater contrast against dimmed background
        let brightened = original.applyingFilter("CIExposureAdjust", parameters: [
            kCIInputEVKey: log2(1.2)
        ])

        // Composite: brightened where mask=white, darkened where mask=black
        let blended: CIImage
        if let maskCI = buildMaskCIImage(originalExtent: original.extent) {
            blended = brightened.applyingFilter("CIBlendWithMask", parameters: [
                kCIInputBackgroundImageKey: darkened,
                kCIInputMaskImageKey: maskCI
            ])
        } else {
            blended = darkened
        }

        // Rotate to portrait
        let oriented = blended.oriented(.right)
        guard let cgImg = ciContext.createCGImage(oriented, from: oriented.extent) else { return nil }

        // Draw bboxes + labels onto the portrait image
        return drawAnnotations(on: cgImg)
    }

    /// Build a mask CIImage at full image resolution using vImage bilinear upsampling.
    /// The proto mask (e.g. 160×160) maps to the model input (640×640) via scaleFit.
    /// We extract the content area (excluding padding), upsample to full res, then wrap as CIImage.
    private func buildMaskCIImage(originalExtent: CGRect) -> CIImage? {
        guard let first = detections.first(where: { $0.mask != nil && !$0.mask!.isEmpty }) else {
            return nil
        }

        let protoW = first.maskWidth   // 160
        let protoH = first.maskHeight  // 160
        let imgW = Int(imageWidth)
        let imgH = Int(imageHeight)

        // Derive scaleFit padding (model size = proto * 4)
        let modelW = Float(protoW * 4)
        let modelH = Float(protoH * 4)
        let s = min(modelW / Float(imgW), modelH / Float(imgH))
        let scaledW = Float(imgW) * s
        let scaledH = Float(imgH) * s
        let protoPadX = Int(round((modelW - scaledW) / 2.0 / 4.0))
        let protoPadY = Int(round((modelH - scaledH) / 2.0 / 4.0))
        let protoContentW = protoW - 2 * protoPadX
        let protoContentH = protoH - 2 * protoPadY

        guard protoContentW > 0, protoContentH > 0 else { return nil }

        // Combine all detection masks at proto resolution (max merge via vDSP)
        // Masks are smooth sigmoid values [0,1], not binary
        var combined = [Float](repeating: 0, count: protoW * protoH)
        for detection in detections {
            guard let mask = detection.mask, mask.count == protoW * protoH else { continue }
            vDSP_vmax(combined, 1, mask, 1, &combined, 1, vDSP_Length(combined.count))
        }

        // Extract content area as float (skip scaleFit padding rows/cols)
        var content = [Float](repeating: 0, count: protoContentW * protoContentH)
        content.withUnsafeMutableBufferPointer { dst in
            combined.withUnsafeBufferPointer { src in
                for y in 0..<protoContentH {
                    let srcOffset = (y + protoPadY) * protoW + protoPadX
                    let dstOffset = y * protoContentW
                    memcpy(dst.baseAddress! + dstOffset, src.baseAddress! + srcOffset,
                           protoContentW * MemoryLayout<Float>.stride)
                }
            }
        }

        // Upsample to full image resolution using vImage bilinear interpolation
        let upsampled = content.withUnsafeBufferPointer { ptr in
            upsampleMask(
                mask: ptr.baseAddress!,
                width: protoContentW,
                height: protoContentH,
                newWidth: imgW,
                newHeight: imgH
            )
        }

        // Convert float [0,1] → UInt8 [0,255], flipping Y axis.
        // The mask data is in pixel-buffer coords (Y=0 at top), but the image
        // CIImage (from CGImage round-trip) composites with Y=0 at bottom.
        // Flipping the mask vertically aligns it with the image in CIImage space.
        let pixelCount = imgW * imgH

        // 1. Scale [0,1] → [0,255] using vDSP
        var scaled = [Float](repeating: 0, count: pixelCount)
        var scaleFactor: Float = 255.0
        vDSP_vsmul(upsampled, 1, &scaleFactor, &scaled, 1, vDSP_Length(pixelCount))

        // 2. Clip to [0, 255]
        var lo: Float = 0, hi: Float = 255
        vDSP_vclip(scaled, 1, &lo, &hi, &scaled, 1, vDSP_Length(pixelCount))

        // 3. Convert Float → UInt8
        var pixels = [UInt8](repeating: 0, count: pixelCount)
        vDSP_vfixu8(scaled, 1, &pixels, 1, vDSP_Length(pixelCount))

        // 4. Flip Y: reverse row order in-place using memcpy row swaps
        var tempRow = [UInt8](repeating: 0, count: imgW)
        for y in 0..<(imgH / 2) {
            let topStart = y * imgW
            let botStart = (imgH - 1 - y) * imgW
            tempRow.withUnsafeMutableBufferPointer { tmp in
                pixels.withUnsafeMutableBufferPointer { pix in
                    memcpy(tmp.baseAddress!, pix.baseAddress! + topStart, imgW)
                    memcpy(pix.baseAddress! + topStart, pix.baseAddress! + botStart, imgW)
                    memcpy(pix.baseAddress! + botStart, tmp.baseAddress!, imgW)
                }
            }
        }

        // Create grayscale mask CIImage at full image resolution
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData),
              let cgMask = CGImage(
                  width: imgW, height: imgH,
                  bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: imgW,
                  space: CGColorSpaceCreateDeviceGray(),
                  bitmapInfo: CGBitmapInfo(rawValue: 0),
                  provider: provider, decode: nil,
                  shouldInterpolate: true, intent: .defaultIntent
              ) else { return nil }

        return CIImage(cgImage: cgMask)
    }

    /// Draw bounding boxes, centered labels, and 3D wireframe directly on the portrait CGImage.
    /// This guarantees pixel-perfect alignment (no SwiftUI layout ambiguity).
    private func drawAnnotations(on cgImg: CGImage) -> UIImage? {
        let w = CGFloat(cgImg.width)
        let h = CGFloat(cgImg.height)
        let size = CGSize(width: w, height: h)

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: size, format: format)

        return renderer.image { ctx in
            // Draw the composited portrait image
            UIImage(cgImage: cgImg).draw(in: CGRect(origin: .zero, size: size))

            let uiCtx = ctx.cgContext

            // Brownish yellow for fused depth, green for LiDAR depth
            let fusedColor = UIColor(red: 0.8, green: 0.6, blue: 0.1, alpha: 0.7)
            let lidarColor = UIColor.green.withAlphaComponent(0.7)

            for (i, detection) in detections.enumerated() {
                let portRect = landscapeToPortrait(detection.boundingBox)
                let extent = i < objectExtents.count ? objectExtents[i] : nil
                let bgColor = (extent?.isLiDAR == true) ? lidarColor : fusedColor
                let strokeColor = (extent?.isLiDAR == true) ? UIColor.green : UIColor(red: 0.8, green: 0.6, blue: 0.1, alpha: 1.0)

                // Bbox
                uiCtx.setStrokeColor(strokeColor.cgColor)
                uiCtx.setLineWidth(4)
                uiCtx.stroke(portRect)

                // 3D wireframe overlay
                if let extent, extent.isLiDAR {
                    if extent.obbCenter != nil {
                        // Point-cloud OBB: always draw
                        drawWireframe(ctx: uiCtx, bbox: detection.boundingBox, extent: extent)
                    } else if extent.depthMin < extent.depthMax {
                        // Fallback 6-point: only if meaningful depth spread
                        let ratio = extent.depthMin / extent.depthMax
                        if ratio <= 0.95 {
                            drawWireframe(ctx: uiCtx, bbox: detection.boundingBox, extent: extent)
                        }
                    }
                }

                // Two-line label: "className 95%" and "2.30 m"
                let line1 = "\(detection.className) \(Int(detection.confidence * 100))%"
                let distStr: String
                if let e = extent {
                    distStr = String(format: "%.2f m", e.medianDepth)
                } else {
                    distStr = "-- m"
                }
                let label = "\(line1)\n\(distStr)"

                let font = UIFont.boldSystemFont(ofSize: 28)
                let paragraphStyle = NSMutableParagraphStyle()
                paragraphStyle.alignment = .center
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: UIColor.white,
                    .paragraphStyle: paragraphStyle,
                ]
                let attrStr = NSAttributedString(string: label, attributes: attrs)
                let labelSize = attrStr.boundingRect(
                    with: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude),
                    options: [.usesLineFragmentOrigin, .usesFontLeading],
                    context: nil
                ).size

                // Label background
                let bgRect = CGRect(
                    x: portRect.midX - labelSize.width / 2 - 6,
                    y: portRect.midY - labelSize.height / 2 - 3,
                    width: labelSize.width + 12,
                    height: labelSize.height + 6
                )
                uiCtx.setFillColor(bgColor.cgColor)
                uiCtx.fill(bgRect)

                // Label text (draw with rect for multiline support)
                let textRect = CGRect(
                    x: portRect.midX - labelSize.width / 2,
                    y: portRect.midY - labelSize.height / 2,
                    width: labelSize.width,
                    height: labelSize.height
                )
                attrStr.draw(in: textRect)
            }
        }
    }

    // MARK: - 3D Wireframe (Gravity-Aligned AABB)

    /// Draw 3D wireframe OBB. Uses pre-computed point-cloud OBB (PCA-aligned) when available,
    /// otherwise falls back to 6-point camera-yaw-aligned method.
    private func drawWireframe(ctx: CGContext, bbox: CGRect, extent: Object3DExtent) {
        // Use pre-computed point-cloud OBB if available, otherwise fall back to 6-point method
        let obbCorners: [SIMD3<Float>]
        if let center = extent.obbCenter, let dims = extent.obbDims, let yaw = extent.obbYaw {
            // 8 corners from PCA-aligned OBB
            let cosY = cos(yaw), sinY = sin(yaw)
            let hw = dims.x / 2, hh = dims.y / 2, hd = dims.z / 2
            // PCA-aligned local axes: X along principal (cos yaw, 0, sin yaw), Y up, Z perpendicular
            let axisX = SIMD3<Float>(cosY, 0, sinY)
            let axisY = SIMD3<Float>(0, 1, 0)
            let axisZ = SIMD3<Float>(-sinY, 0, cosY)
            obbCorners = [
                center - axisX * hw - axisY * hh - axisZ * hd, // 0
                center + axisX * hw - axisY * hh - axisZ * hd, // 1
                center + axisX * hw - axisY * hh + axisZ * hd, // 2
                center - axisX * hw - axisY * hh + axisZ * hd, // 3
                center - axisX * hw + axisY * hh - axisZ * hd, // 4
                center + axisX * hw + axisY * hh - axisZ * hd, // 5
                center + axisX * hw + axisY * hh + axisZ * hd, // 6
                center - axisX * hw + axisY * hh + axisZ * hd, // 7
            ]
        } else {
            // Fallback: 6-point camera-yaw-aligned method
            let corners2D = [
                CGPoint(x: bbox.minX, y: bbox.minY),
                CGPoint(x: bbox.maxX, y: bbox.minY),
                CGPoint(x: bbox.maxX, y: bbox.maxY),
                CGPoint(x: bbox.minX, y: bbox.maxY),
            ]
            var worldPoints: [SIMD3<Float>] = corners2D.map {
                unprojectToWorld($0, depth: extent.medianDepth)
            }
            let center2D = CGPoint(x: bbox.midX, y: bbox.midY)
            worldPoints.append(unprojectToWorld(center2D, depth: extent.depthMin))
            worldPoints.append(unprojectToWorld(center2D, depth: extent.depthMax))

            let camForward3 = SIMD3<Float>(cameraTransform.columns.2.x,
                                            cameraTransform.columns.2.y,
                                            cameraTransform.columns.2.z)
            let horizForward = normalize(SIMD3<Float>(-camForward3.x, 0, -camForward3.z))
            let up = SIMD3<Float>(0, 1, 0)
            let right = normalize(cross(horizForward, up))

            var minR: Float = .greatestFiniteMagnitude, maxR: Float = -.greatestFiniteMagnitude
            var minU: Float = .greatestFiniteMagnitude, maxU: Float = -.greatestFiniteMagnitude
            var minF: Float = .greatestFiniteMagnitude, maxF: Float = -.greatestFiniteMagnitude
            for (i, p) in worldPoints.enumerated() {
                let r = dot(p, right); let u = dot(p, up); let f = dot(p, horizForward)
                if i < 4 {
                    minR = min(minR, r); maxR = max(maxR, r)
                    minU = min(minU, u); maxU = max(maxU, u)
                }
                minF = min(minF, f); maxF = max(maxF, f)
            }
            obbCorners = [
                right * minR + up * minU + horizForward * minF,
                right * maxR + up * minU + horizForward * minF,
                right * maxR + up * minU + horizForward * maxF,
                right * minR + up * minU + horizForward * maxF,
                right * minR + up * maxU + horizForward * minF,
                right * maxR + up * maxU + horizForward * minF,
                right * maxR + up * maxU + horizForward * maxF,
                right * minR + up * maxU + horizForward * maxF,
            ]
        }

        // Project to portrait image space
        let invTransform = cameraTransform.inverse
        let projected: [CGPoint?] = obbCorners.map { wp in
            guard let lp = projectToImage(wp, invTransform: invTransform) else { return nil }
            return CGPoint(x: lp.y, y: lp.x) // landscape → portrait
        }

        // 12 edges of cuboid
        let edges = [
            (0,1), (1,2), (2,3), (3,0), // bottom
            (4,5), (5,6), (6,7), (7,4), // top
            (0,4), (1,5), (2,6), (3,7), // vertical
        ]

        ctx.setStrokeColor(UIColor.cyan.withAlphaComponent(0.6).cgColor)
        ctx.setLineDash(phase: 0, lengths: [8, 6])
        ctx.setLineWidth(2)

        for (i, j) in edges {
            guard let p1 = projected[i], let p2 = projected[j] else { continue }
            ctx.move(to: p1)
            ctx.addLine(to: p2)
        }
        ctx.strokePath()

        ctx.setLineDash(phase: 0, lengths: [])
    }

    /// Unproject a landscape image point at a given depth to world space.
    private func unprojectToWorld(_ imagePoint: CGPoint, depth: Float) -> SIMD3<Float> {
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        let x = (Float(imagePoint.x) - cx) / fx * depth
        let y = (Float(imagePoint.y) - cy) / fy * depth
        let z = -depth
        let camPt = SIMD4<Float>(x, y, z, 1)
        let worldPt = cameraTransform * camPt
        return SIMD3(worldPt.x, worldPt.y, worldPt.z)
    }

    /// Project a world point back to landscape image coordinates (nil if behind camera).
    private func projectToImage(_ worldPoint: SIMD3<Float>, invTransform: simd_float4x4? = nil) -> CGPoint? {
        let inv = invTransform ?? cameraTransform.inverse
        let camPt = inv * SIMD4<Float>(worldPoint.x, worldPoint.y, worldPoint.z, 1)
        guard camPt.z < -0.01 else { return nil }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        let px = fx * (camPt.x / -camPt.z) + cx
        let py = fy * (camPt.y / -camPt.z) + cy
        return CGPoint(x: CGFloat(px), y: CGFloat(py))
    }

    // MARK: - Debug View (LiDAR Depth + Mask Contours)

    /// Build a debug image showing LiDAR depth heatmap with bounding boxes and mask contours
    /// for all detections. Green contour = original mask edge, Red = eroded mask edge.
    private func buildDebugImage() -> UIImage? {
        guard let lidar = lidarSnapshot else { return nil }

        let lw = lidar.width   // typically 256
        let lh = lidar.height  // typically 192

        // Find depth range for normalization
        var minDepth: Float = .greatestFiniteMagnitude
        var maxDepth: Float = 0
        for i in 0..<(lw * lh) {
            let d = lidar.depthValues[i]
            if d > 0.05 && d < 10.0 && d.isFinite {
                if d < minDepth { minDepth = d }
                if d > maxDepth { maxDepth = d }
            }
        }
        guard maxDepth > minDepth else { return nil }
        let depthRange = maxDepth - minDepth

        // Build RGBA pixel buffer (landscape orientation)
        var rgba = [UInt8](repeating: 0, count: lw * lh * 4)
        for i in 0..<(lw * lh) {
            let d = lidar.depthValues[i]
            let offset = i * 4
            if d > 0.05 && d < 10.0 && d.isFinite {
                let norm = (d - minDepth) / depthRange
                let (r, g, b) = debugDepthToColor(norm)
                rgba[offset] = r
                rgba[offset + 1] = g
                rgba[offset + 2] = b
            }
            rgba[offset + 3] = 255
        }

        let scaleX = Float(lw) / Float(imageWidth)
        let scaleY = Float(lh) / Float(imageHeight)

        // Draw each detection's bbox and mask contours.
        // IMPORTANT: YOLO bbox/mask Y is flipped (Y=0 at scene bottom) because
        // toBGRAData() renders through CIImage which flips Y. LiDAR depth Y=0
        // is at scene top. We must flip bbox Y when mapping to LiDAR coords.
        let imgH = Float(imageHeight)
        for det in detections {
            let bbox = det.boundingBox
            let lx0 = max(0, Int(Float(bbox.minX) * scaleX))
            let ly0 = max(0, Int((imgH - Float(bbox.maxY)) * scaleY))
            let lx1 = min(lw - 1, Int(Float(bbox.maxX) * scaleX))
            let ly1 = min(lh - 1, Int((imgH - Float(bbox.minY)) * scaleY))

            // Draw bounding box outline in green
            debugDrawRect(&rgba, width: lw, height: lh,
                          x0: lx0, y0: ly0, x1: lx1, y1: ly1,
                          r: 0, g: 255, b: 0)

            guard let mask = det.mask, !mask.isEmpty else { continue }

            let protoW = det.maskWidth
            let protoH = det.maskHeight

            // ScaleFit padding (same formula as computeObject3DExtent / buildMaskCIImage)
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
            let lidarToProtoX = protoContentW / Float(lw)
            let lidarToProtoY = protoContentH / Float(lh)

            // Compute eroded mask (same parameters as ContentView.computeObject3DExtent)
            let erosionImagePx = 0.05 * min(Float(bbox.width), Float(bbox.height))
            let imageToProtoScale = protoContentW / Float(imageWidth)
            let erosionRadius = min(3, max(1, Int(round(erosionImagePx * imageToProtoScale))))
            let eroded = erodeMask(mask, width: protoW, height: protoH, radius: erosionRadius)

            guard lx0 <= lx1, ly0 <= ly1 else { continue }

            for ly in ly0...ly1 {
                for lx in lx0...lx1 {
                    // Proto mask Y is flipped (YOLO input was Y-flipped by CIContext.render).
                    // LiDAR ly is Y=0-at-top; convert to flipped Y for proto sampling.
                    let protoX = Float(lx) * lidarToProtoX + protoPadX
                    let protoY = Float(lh - 1 - ly) * lidarToProtoY + protoPadY

                    let origVal = debugSampleMask(mask, width: protoW, height: protoH, px: protoX, py: protoY)
                    let erodVal = debugSampleMask(eroded, width: protoW, height: protoH, px: protoX, py: protoY)

                    // Contour: pixel >= 0.5 with any 4-neighbor < 0.5
                    let isOrigContour = origVal >= 0.5 && debugIsEdge(
                        mask, width: protoW, height: protoH, lx: lx, ly: ly, lh: lh,
                        lidarToProtoX: lidarToProtoX, lidarToProtoY: lidarToProtoY,
                        protoPadX: protoPadX, protoPadY: protoPadY)
                    let isErodContour = erodVal >= 0.5 && debugIsEdge(
                        eroded, width: protoW, height: protoH, lx: lx, ly: ly, lh: lh,
                        lidarToProtoX: lidarToProtoX, lidarToProtoY: lidarToProtoY,
                        protoPadX: protoPadX, protoPadY: protoPadY)

                    let offset = (ly * lw + lx) * 4
                    if isErodContour {
                        rgba[offset] = 255; rgba[offset + 1] = 50; rgba[offset + 2] = 50
                    } else if isOrigContour {
                        rgba[offset] = 50; rgba[offset + 1] = 255; rgba[offset + 2] = 50
                    }
                }
            }
        }

        // Manual 90° CW rotation to portrait: landscape (lw × lh) → portrait (lh × lw)
        // Derived from CIImage(cgImage:).oriented(.right).createCGImage chain:
        //   portrait(px, py) ← landscape(py, lh - 1 - px)
        // This bypasses CIImage entirely, avoiding all Y-flip confusion.
        let portW = lh
        let portH = lw
        var portraitRgba = [UInt8](repeating: 0, count: portW * portH * 4)
        for py in 0..<portH {
            for px in 0..<portW {
                let lx = py
                let ly = lh - 1 - px
                let srcOff = (ly * lw + lx) * 4
                let dstOff = (py * portW + px) * 4
                portraitRgba[dstOff]     = rgba[srcOff]
                portraitRgba[dstOff + 1] = rgba[srcOff + 1]
                portraitRgba[dstOff + 2] = rgba[srcOff + 2]
                portraitRgba[dstOff + 3] = rgba[srcOff + 3]
            }
        }

        guard let provider = CGDataProvider(data: Data(portraitRgba) as CFData),
              let cgPortrait = CGImage(
                  width: portW, height: portH,
                  bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: portW * 4,
                  space: CGColorSpaceCreateDeviceRGB(),
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                  provider: provider, decode: nil,
                  shouldInterpolate: false, intent: .defaultIntent
              ) else { return nil }

        return UIImage(cgImage: cgPortrait)
    }

    // MARK: - Debug Helpers

    private func debugSampleMask(_ m: [Float], width: Int, height: Int, px: Float, py: Float) -> Float {
        let x0 = max(0, Int(px))
        let y0 = max(0, Int(py))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let fx = max(0, px - Float(x0))
        let fy = max(0, py - Float(y0))
        return m[y0 * width + x0] * (1 - fx) * (1 - fy) + m[y0 * width + x1] * fx * (1 - fy)
             + m[y1 * width + x0] * (1 - fx) * fy + m[y1 * width + x1] * fx * fy
    }

    private func debugIsEdge(_ mask: [Float], width: Int, height: Int,
                             lx: Int, ly: Int, lh: Int,
                             lidarToProtoX: Float, lidarToProtoY: Float,
                             protoPadX: Float, protoPadY: Float) -> Bool {
        for (nx, ny) in [(lx - 1, ly), (lx + 1, ly), (lx, ly - 1), (lx, ly + 1)] {
            let px = Float(nx) * lidarToProtoX + protoPadX
            let py = Float(lh - 1 - ny) * lidarToProtoY + protoPadY
            if debugSampleMask(mask, width: width, height: height, px: px, py: py) < 0.5 { return true }
        }
        return false
    }

    private func debugDrawRect(_ rgba: inout [UInt8], width: Int, height: Int,
                               x0: Int, y0: Int, x1: Int, y1: Int,
                               r: UInt8, g: UInt8, b: UInt8) {
        func set(_ px: Int, _ py: Int) {
            guard px >= 0, px < width, py >= 0, py < height else { return }
            let o = (py * width + px) * 4
            rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = 255
        }
        for x in x0...x1 { set(x, y0); set(x, y1) }
        for y in y0...y1 { set(x0, y); set(x1, y) }
    }

    /// Map normalized depth [0,1] to a turbo-like heatmap (blue -> cyan -> green -> yellow -> red).
    private func debugDepthToColor(_ normalized: Float) -> (r: UInt8, g: UInt8, b: UInt8) {
        let t = max(0, min(1, normalized))
        let r = UInt8(max(0, min(255, 255 * (1.5 - abs(t - 0.75) * 4))))
        let g = UInt8(max(0, min(255, 255 * (1.5 - abs(t - 0.5) * 4))))
        let b = UInt8(max(0, min(255, 255 * (1.5 - abs(t - 0.25) * 4))))
        return (r, g, b)
    }
}
