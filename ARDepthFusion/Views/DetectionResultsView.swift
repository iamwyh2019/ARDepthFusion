import SwiftUI
import CoreImage
import CoreImage.CIFilterBuiltins
import Accelerate

struct DetectionResultsView: View {
    let capturedCIImage: CIImage
    let detections: [DetectedObject]
    let imageWidth: CGFloat
    let imageHeight: CGFloat
    let elapsedMs: Double
    let onPlaceEffect: (ParticleEffectType, DetectedObject) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var composited: UIImage?
    @State private var selectedDetection: DetectedObject?
    @State private var statusText = ""

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

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
        let ciContext = CIContext(options: [.useSoftwareRenderer: false])
        let original = capturedCIImage

        // CIExposureAdjust preserves hue. EV = log2(0.3) ≈ -1.74 → 0.3x brightness
        let darkened = original.applyingFilter("CIExposureAdjust", parameters: [
            kCIInputEVKey: -1.74
        ])

        // Composite: original where mask=white, darkened where mask=black
        let blended: CIImage
        if let maskCI = buildMaskCIImage(originalExtent: original.extent) {
            blended = original.applyingFilter("CIBlendWithMask", parameters: [
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
        for y in 0..<protoContentH {
            for x in 0..<protoContentW {
                let idx = (y + protoPadY) * protoW + (x + protoPadX)
                content[y * protoContentW + x] = combined[idx]
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
        var pixels = [UInt8](repeating: 0, count: imgW * imgH)
        for y in 0..<imgH {
            let srcY = imgH - 1 - y
            for x in 0..<imgW {
                pixels[y * imgW + x] = UInt8(min(max(upsampled[srcY * imgW + x] * 255.0, 0), 255))
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

    /// Draw bounding boxes and centered labels directly on the portrait CGImage.
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

            // DEBUG: corner markers to verify coordinate system
            // RED = origin (0,0), BLUE = (maxX, 0), YELLOW = (0, maxY)
            let markerSize: CGFloat = 60
            uiCtx.setFillColor(UIColor.red.cgColor)
            uiCtx.fill(CGRect(x: 0, y: 0, width: markerSize, height: markerSize))
            uiCtx.setFillColor(UIColor.blue.cgColor)
            uiCtx.fill(CGRect(x: w - markerSize, y: 0, width: markerSize, height: markerSize))
            uiCtx.setFillColor(UIColor.yellow.cgColor)
            uiCtx.fill(CGRect(x: 0, y: h - markerSize, width: markerSize, height: markerSize))

            print("[DEBUG] portrait CGImage: \(Int(w))×\(Int(h)), imageWidth=\(imageWidth), imageHeight=\(imageHeight)")

            for detection in detections {
                let portRect = landscapeToPortrait(detection.boundingBox)
                print("[DEBUG] \(detection.className): landscape bbox=\(detection.boundingBox) → portrait=\(portRect)")

                // Green bbox
                uiCtx.setStrokeColor(UIColor.green.cgColor)
                uiCtx.setLineWidth(4)
                uiCtx.stroke(portRect)

                // Centered label
                let label = "\(detection.className) \(Int(detection.confidence * 100))%"
                let font = UIFont.boldSystemFont(ofSize: 28)
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: font,
                    .foregroundColor: UIColor.white,
                ]
                let labelSize = (label as NSString).size(withAttributes: attrs)

                // Label background
                let bgRect = CGRect(
                    x: portRect.midX - labelSize.width / 2 - 6,
                    y: portRect.midY - labelSize.height / 2 - 3,
                    width: labelSize.width + 12,
                    height: labelSize.height + 6
                )
                uiCtx.setFillColor(UIColor.green.withAlphaComponent(0.7).cgColor)
                uiCtx.fill(bgRect)

                // Label text
                let textPoint = CGPoint(
                    x: portRect.midX - labelSize.width / 2,
                    y: portRect.midY - labelSize.height / 2
                )
                (label as NSString).draw(at: textPoint, withAttributes: attrs)
            }
        }
    }
}
