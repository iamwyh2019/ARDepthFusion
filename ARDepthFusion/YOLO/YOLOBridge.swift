import Foundation
import Vision
import UIKit

// MARK: - Detection Result Types

struct YOLODetection: Sendable {
    let classIndex: Int
    let className: String
    let confidence: Float
    let boxX1: Int
    let boxY1: Int
    let boxX2: Int
    let boxY2: Int
    let centroidX: Int
    let centroidY: Int
}

struct YOLODetectionResult: Sendable {
    let detections: [YOLODetection]
    let timestamp: UInt64
}

// MARK: - Callback (Swift closure, no C ABI needed)

typealias YOLOCallback = @Sendable (YOLODetectionResult) -> Void

// MARK: - Global State

nonisolated(unsafe) var predictor: YOLOPredictor? = nil
nonisolated(unsafe) var yoloCallback: YOLOCallback? = nil

// MARK: - API Functions

nonisolated func RegisterYOLOCallback(callback: @escaping YOLOCallback) {
    yoloCallback = callback
}

nonisolated func InitializeYOLO(
    modelName: String,
    confidenceThreshold: Float,
    iouThreshold: Float,
    scaleMethod: String
) -> Bool {
    predictor = YOLOPredictor(
        modelName: modelName,
        confidanceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold,
        scaleMethod: scaleMethod
    )
    return predictor != nil
}

nonisolated func RunYOLO_Byte(
    imageData: UnsafePointer<UInt8>,
    width: Int,
    height: Int,
    timestamp: UInt64 = 0,
    scaleX: Float = 1.0,
    scaleY: Float = 1.0
) {
    guard let predictor = predictor else {
        NSLog("Error: YOLOPredictor not initialized.")
        return
    }

    guard let cvPixelBuffer = bytesToCVPixelBuffer(data: imageData, width: width, height: height) else {
        NSLog("Error: Failed to convert image data.")
        return
    }

    predictor.predict(
        cvPixelBuffer: cvPixelBuffer,
        timestamp: timestamp == 0 ? getCurrentTimestamp() : timestamp,
        scaleX: scaleX,
        scaleY: scaleY
    )
}
