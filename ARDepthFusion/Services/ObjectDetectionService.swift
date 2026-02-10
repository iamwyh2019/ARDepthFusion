import ARKit
import YOLOUnity

private func yoloCallbackHandler(
    _ classCount: Int32,
    _ classIndices: UnsafePointer<Int32>,
    _ classNamesRaw: UnsafePointer<UInt8>,
    _ namesByteLen: Int32,
    _ confidences: UnsafePointer<Float>,
    _ boxes: UnsafePointer<Int32>,
    _ centroids: UnsafePointer<Int32>,
    _ detectionCount: Int32,
    _ maskSizes: UnsafePointer<Int32>,
    _ maskCount: Int32,
    _ maskData: UnsafePointer<Int32>,
    _ timestamp: UInt64
) {
    let count = Int(detectionCount)
    guard count > 0 else {
        ObjectDetectionService.shared.deliverResults([])
        return
    }

    // Parse class names from null-separated UTF-8
    var classNames: [String] = []
    let namesLen = Int(namesByteLen)
    var start = 0
    for i in 0..<namesLen {
        if classNamesRaw[i] == 0 {
            let nameData = Data(bytes: classNamesRaw.advanced(by: start), count: i - start)
            classNames.append(String(data: nameData, encoding: .utf8) ?? "unknown")
            start = i + 1
        }
    }

    var objects: [DetectedObject] = []
    for i in 0..<count {
        let classIdx = Int(classIndices[i])
        let name = classIdx < classNames.count ? classNames[classIdx] : "class_\(classIdx)"
        let conf = confidences[i]

        // Boxes are XYXY format (x1, y1, x2, y2) as Int32
        let x1 = CGFloat(boxes[i * 4 + 0])
        let y1 = CGFloat(boxes[i * 4 + 1])
        let x2 = CGFloat(boxes[i * 4 + 2])
        let y2 = CGFloat(boxes[i * 4 + 3])
        let bbox = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)

        // Centroids are (cx, cy) pairs as Int32
        let cx = CGFloat(centroids[i * 2 + 0])
        let cy = CGFloat(centroids[i * 2 + 1])

        objects.append(DetectedObject(
            className: name,
            confidence: conf,
            boundingBox: bbox,
            centroid: CGPoint(x: cx, y: cy)
        ))
    }

    ObjectDetectionService.shared.deliverResults(objects)
}

@Observable
final class ObjectDetectionService: @unchecked Sendable {
    nonisolated(unsafe) static let shared = ObjectDetectionService()

    private var isInitialized = false
    private let lock = NSLock()
    private var pendingContinuation: CheckedContinuation<[DetectedObject], Never>?
    private var retainedImageData: Data?

    private init() {}

    func initialize() {
        guard !isInitialized else { return }
        let success = InitializeYOLO(
            modelName: "yolo11l_seg",
            confidenceThreshold: 0.5,
            iouThreshold: 0.5,
            scaleMethod: "scaleFit"
        )
        if success {
            RegisterYOLOCallback(callback: yoloCallbackHandler)
            isInitialized = true
            print("YOLO initialized successfully")
        } else {
            print("YOLO initialization failed")
        }
    }

    func detect(frame: ARFrame) async -> [DetectedObject] {
        if !isInitialized { initialize() }

        guard let bgra = frame.capturedImage.toBGRAData() else {
            return []
        }

        return await withCheckedContinuation { continuation in
            lock.lock()
            pendingContinuation = continuation
            retainedImageData = bgra.data
            lock.unlock()

            bgra.data.withUnsafeBytes { ptr in
                guard let baseAddress = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else {
                    self.deliverResults([])
                    return
                }
                RunYOLO_Byte(
                    imageData: baseAddress,
                    width: bgra.width,
                    height: bgra.height,
                    timestamp: UInt64(Date().timeIntervalSince1970 * 1000)
                )
            }
        }
    }

    nonisolated func deliverResults(_ objects: [DetectedObject]) {
        lock.lock()
        let continuation = pendingContinuation
        pendingContinuation = nil
        retainedImageData = nil
        lock.unlock()

        continuation?.resume(returning: objects)
    }
}
