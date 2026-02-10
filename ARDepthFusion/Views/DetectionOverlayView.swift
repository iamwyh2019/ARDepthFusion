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
    // imageX -> maps to bottom of screen, imageY -> maps to right of screen
    // screenX = imageY / imageHeight * viewWidth
    // screenY = (1 - imageX / imageWidth) * viewHeight
    private func imageToScreen(_ rect: CGRect, viewSize: CGSize) -> CGRect {
        let x1 = rect.minY / imageHeight * viewSize.width
        let y1 = (1 - rect.maxX / imageWidth) * viewSize.height
        let x2 = rect.maxY / imageHeight * viewSize.width
        let y2 = (1 - rect.minX / imageWidth) * viewSize.height

        return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
    }
}
