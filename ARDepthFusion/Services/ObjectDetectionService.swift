import Combine
import CoreGraphics
import Foundation

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

final class ObjectDetectionService: ObservableObject, @unchecked Sendable {
    nonisolated static let shared = ObjectDetectionService()

    private var isInitialized = false
    private let lock = NSLock()
    private nonisolated(unsafe) var pendingContinuation: CheckedContinuation<[DetectedObject], Never>?
    private nonisolated(unsafe) var retainedImageData: Data?

    private init() {}

    /// Initialize YOLO model. Returns nil on success, or an error message on failure.
    func initialize() async -> String? {
        guard !isInitialized else { return nil }
        let success = await Task.detached(priority: .userInitiated) {
            InitializeYOLO(
                modelName: "yolo11l-seg",
                confidenceThreshold: 0.75,
                iouThreshold: 0.5,
                scaleMethod: "scaleFit"
            )
        }.value
        if success {
            RegisterYOLOCallback(callback: yoloResultHandler)
            isInitialized = true
            print("YOLO initialized successfully")
            return nil
        } else {
            let msg = "Failed to load YOLO model 'yolo11l-seg'. Check that the compiled model (.mlmodelc) is included in the app bundle."
            print(msg)
            return msg
        }
    }

    func detect(bgraData: Data, width: Int, height: Int) async -> [DetectedObject] {
        if !isInitialized { _ = await initialize() }
        guard isInitialized else { return [] }

        let nsData = bgraData as NSData
        let baseAddress = nsData.bytes.assumingMemoryBound(to: UInt8.self)

        return await withCheckedContinuation { continuation in
            lock.lock()
            pendingContinuation = continuation
            retainedImageData = bgraData
            lock.unlock()

            RunYOLO_Byte(
                imageData: baseAddress,
                width: width,
                height: height,
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
