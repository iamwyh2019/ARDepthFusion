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
}
