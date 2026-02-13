import CoreImage
import CoreVideo

private let sharedCIContext = CIContext(options: [.useSoftwareRenderer: false])

extension CVPixelBuffer {
    func toBGRAData() -> (data: Data, width: Int, height: Int)? {
        let ciImage = CIImage(cvPixelBuffer: self)
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)

        let bytesPerRow = width * 4
        var bgraData = Data(count: bytesPerRow * height)

        bgraData.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            sharedCIContext.render(
                ciImage,
                toBitmap: baseAddress,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .BGRA8,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )
        }

        return (bgraData, width, height)
    }

    /// Rotate landscape-right capture to portrait, then extract BGRA bytes.
    /// Uses `.oriented(.left)` because the render→YOLO pipeline includes two Y-flips
    /// (render toBitmap + bytesToCVPixelBuffer flipY:true), making `.left` produce the
    /// correct upright portrait after the full chain.
    /// Returns portrait dimensions (e.g. 1440×1920 from a 1920×1440 landscape buffer).
    func toPortraitBGRAData() -> (data: Data, width: Int, height: Int)? {
        let ciImage = CIImage(cvPixelBuffer: self).oriented(.left)
        let pW = Int(ciImage.extent.width)   // 1440 (landscape height)
        let pH = Int(ciImage.extent.height)  // 1920 (landscape width)
        let bytesPerRow = pW * 4
        var bgraData = Data(count: bytesPerRow * pH)

        bgraData.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }
            sharedCIContext.render(
                ciImage,
                toBitmap: baseAddress,
                rowBytes: bytesPerRow,
                bounds: ciImage.extent,
                format: .BGRA8,
                colorSpace: CGColorSpaceCreateDeviceRGB()
            )
        }

        return (bgraData, pW, pH)
    }
}
