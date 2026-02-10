import ARKit

nonisolated private func yoloResultHandler(_ result: YOLODetectionResult) {
    let objects = result.detections.map { det in
        DetectedObject(
            className: det.className,
            confidence: det.confidence,
            boundingBox: CGRect(
                x: CGFloat(det.boxX1),
                y: CGFloat(det.boxY1),
                width: CGFloat(det.boxX2 - det.boxX1),
                height: CGFloat(det.boxY2 - det.boxY1)
            ),
            centroid: CGPoint(x: CGFloat(det.centroidX), y: CGFloat(det.centroidY)),
            mask: det.mask.isEmpty ? nil : det.mask,
            maskWidth: det.maskWidth,
            maskHeight: det.maskHeight
        )
    }

    ObjectDetectionService.shared.deliverResults(objects)
}

@Observable
final class ObjectDetectionService: @unchecked Sendable {
    nonisolated static let shared = ObjectDetectionService()

    private var isInitialized = false
    private let lock = NSLock()
    @ObservationIgnored private nonisolated(unsafe) var pendingContinuation: CheckedContinuation<[DetectedObject], Never>?
    @ObservationIgnored private nonisolated(unsafe) var retainedImageData: Data?

    private init() {}

    func initialize() async {
        guard !isInitialized else { return }
        let success = await Task.detached(priority: .userInitiated) {
            InitializeYOLO(
                modelName: "yolo11l_seg",
                confidenceThreshold: 0.7,
                iouThreshold: 0.5,
                scaleMethod: "scaleFit"
            )
        }.value
        if success {
            RegisterYOLOCallback(callback: yoloResultHandler)
            isInitialized = true
            print("YOLO initialized successfully")
        } else {
            print("YOLO initialization failed")
        }
    }

    func detect(frame: ARFrame) async -> [DetectedObject] {
        if !isInitialized { await initialize() }

        guard let bgra = frame.capturedImage.toBGRAData() else {
            return []
        }

        let nsData = bgra.data as NSData
        let baseAddress = nsData.bytes.assumingMemoryBound(to: UInt8.self)

        return await withCheckedContinuation { continuation in
            lock.lock()
            pendingContinuation = continuation
            retainedImageData = bgra.data
            lock.unlock()

            RunYOLO_Byte(
                imageData: baseAddress,
                width: bgra.width,
                height: bgra.height,
                timestamp: UInt64(Date().timeIntervalSince1970 * 1000)
            )
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
