import SwiftUI

struct DetectionOverlayView: View {
    let detections: [DetectedObject]
    let imageWidth: CGFloat
    let imageHeight: CGFloat
    let onTap: (DetectedObject) -> Void

    var body: some View {
        GeometryReader { geometry in
            ForEach(detections) { detection in
                let screenRect = imageToScreen(
                    detection.boundingBox,
                    viewSize: geometry.size
                )

                Rectangle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: screenRect.width, height: screenRect.height)
                    .overlay(alignment: .topLeading) {
                        Text("\(detection.className) \(Int(detection.confidence * 100))%")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.7))
                    }
                    .position(
                        x: screenRect.midX,
                        y: screenRect.midY
                    )
                    .onTapGesture {
                        onTap(detection)
                    }
            }
        }
    }

    // Camera captures in landscape-right (1920x1440), displayed in portrait.
    // Step 1: Rotate 90° CW → rotated image is imageHeight x imageWidth (1440x1920)
    // Step 2: ARView does aspect-fill into viewSize, so we must compute scale + crop offset.
    private func imageToScreen(_ rect: CGRect, viewSize: CGSize) -> CGRect {
        // After rotation: rotated dimensions
        let rotW = imageHeight  // 1440
        let rotH = imageWidth   // 1920

        // Aspect-fill: scale to fill entire view, crop the excess
        let scale = max(viewSize.width / rotW, viewSize.height / rotH)
        let offsetX = (rotW * scale - viewSize.width) / 2
        let offsetY = (rotH * scale - viewSize.height) / 2

        // Rotation mapping: landscape (imgX, imgY) → portrait
        // screenX ∝ (imgH - imgY), screenY ∝ imgX
        let rotX1 = imageHeight - rect.maxY
        let rotY1 = rect.minX
        let rotX2 = imageHeight - rect.minY
        let rotY2 = rect.maxX

        let sx1 = rotX1 * scale - offsetX
        let sy1 = rotY1 * scale - offsetY
        let sx2 = rotX2 * scale - offsetX
        let sy2 = rotY2 * scale - offsetY

        return CGRect(x: sx1, y: sy1, width: sx2 - sx1, height: sy2 - sy1)
    }
}
