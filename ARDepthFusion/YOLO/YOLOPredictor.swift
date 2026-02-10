import CoreML
import Vision
import UIKit
import Accelerate

class YOLOPredictor: @unchecked Sendable {
    nonisolated let model: MLModel
    nonisolated let detector: VNCoreMLModel
    nonisolated let confidenceThreshold: Float
    nonisolated let iouThreshold: Float
    nonisolated let modelWidth: Int
    nonisolated let modelHeight: Int
    nonisolated let classNames: [Int: String]
    nonisolated(unsafe) var visionRequest: YOLORequest!

    nonisolated init?(
        modelName: String,
        confidanceThreshold: Float = 0.5,
        iouThreshold: Float = 0.5,
        scaleMethod: String = "scaleFill"
    ) {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        guard let model: MLModel = {
            switch modelName {
            case "yolo11l_seg":
                return try? yolo11l_seg(configuration: config).model
            default:
                NSLog("Error: Unknown model name '\(modelName)'.")
                return nil
            }
        }() else {
            return nil
        }

        guard let detector = try? VNCoreMLModel(for: model) else {
            NSLog("Error: Failed to initialize the detector.")
            return nil
        }

        self.model = model
        self.detector = detector
        self.detector.featureProvider = ThresholdProvider()
        self.confidenceThreshold = confidanceThreshold
        self.iouThreshold = iouThreshold

        (self.modelWidth, self.modelHeight, self.classNames) = parseModelSizeAndNames(model: model)

        let request = YOLORequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
        })

        switch scaleMethod {
        case "scaleFit":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        case "scaleFill":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        case "centerCrop":
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.centerCrop
        default:
            NSLog("Cannot parse scaleMethod: \(scaleMethod), defaulting to scaleFit")
            request.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFit
        }

        visionRequest = request

        NSLog("Initialized model \(modelName) with scaleMethod=\(scaleMethod), score threshold=\(confidenceThreshold), iou threshold=\(iouThreshold), Model width=\(modelWidth), height=\(modelHeight), numClasses=\(classNames.count)")
    }


    nonisolated func predict(cgImage: CGImage, timestamp: UInt64, scaleX: Float = 1.0, scaleY: Float = 1.0) {
        visionRequest.userData["timestamp"] = timestamp
        visionRequest.userData["originalWidth"] = cgImage.width
        visionRequest.userData["originalHeight"] = cgImage.height
        visionRequest.userData["scaleX"] = scaleX
        visionRequest.userData["scaleY"] = scaleY

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        do {
            try handler.perform([visionRequest])
        } catch {
            NSLog("Prediction failed: \(error.localizedDescription)")
        }
    }

    nonisolated func predict(cvPixelBuffer: CVPixelBuffer, timestamp: UInt64, scaleX: Float = 1.0, scaleY: Float = 1.0) {
        visionRequest.userData["timestamp"] = timestamp
        visionRequest.userData["originalWidth"] = CVPixelBufferGetWidth(cvPixelBuffer)
        visionRequest.userData["originalHeight"] = CVPixelBufferGetHeight(cvPixelBuffer)
        visionRequest.userData["scaleX"] = scaleX
        visionRequest.userData["scaleY"] = scaleY

        let handler = VNImageRequestHandler(cvPixelBuffer: cvPixelBuffer, options: [:])
        do {
            try handler.perform([visionRequest])
        } catch {
            NSLog("Prediction failed: \(error.localizedDescription)")
        }
    }

    nonisolated func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.global(qos: .userInitiated).async {
            autoreleasepool {
                if let error = error {
                    NSLog("Error in processing observations: \(error.localizedDescription)")
                    return
                }

                guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                      results.count >= 2 else {
                    NSLog("Error: Insufficient results. Found \(request.results?.count ?? 0) results.")
                    return
                }

                guard let array0 = results[0].featureValue.multiArrayValue,
                      let array1 = results[1].featureValue.multiArrayValue else {
                    NSLog("Error: Could not get multi-arrays from results")
                    return
                }

                let array0Rank = array0.shape.count
                let array1Rank = array1.shape.count

                let (boxes, masks) = (array0Rank == 3 && array1Rank == 4) ? (array0, array1) :
                (array0Rank == 4 && array1Rank == 3) ? (array1, array0) :
                {
                    NSLog("Error: Unexpected array dimensions. Array 0: \(array0.shape), Array 1: \(array1.shape)")
                    return (array0, array1)
                }()

                guard let yoloRequest = request as? YOLORequest,
                      let originalWidth = yoloRequest.userData["originalWidth"] as? Int,
                      let originalHeight = yoloRequest.userData["originalHeight"] as? Int,
                      let timestamp = yoloRequest.userData["timestamp"] as? UInt64,
                      let scaleX = yoloRequest.userData["scaleX"] as? Float,
                      let scaleY = yoloRequest.userData["scaleY"] as? Float
                else {
                    NSLog("Missing image properties")
                    return
                }

                let coordinateRestorer = getCoordinateRestorer(
                    originalSize: (Float(originalWidth), Float(originalHeight)),
                    targetSize: (Float(self.modelWidth), Float(self.modelHeight)),
                    option: self.visionRequest.imageCropAndScaleOption
                )

                let numMasks = masks.shape[1].intValue
                let numClasses = boxes.shape[1].intValue - 4 - numMasks

                let boxPredictions: [BoxPrediction] = parseBoundingBoxes(
                    multiArray: boxes,
                    numClasses: numClasses,
                    confidenceThreshold: self.confidenceThreshold
                )

                // Apply NMS per class
                let groupedPredictions = Dictionary(grouping: boxPredictions) { $0.classIndex }
                var nmsPredictions: [BoxPrediction] = []
                nmsPredictions.reserveCapacity(100)
                for (_, predictions) in groupedPredictions {
                    nmsPredictions.append(
                        contentsOf: nonMaximumSuppression(
                            predictions: predictions,
                            iouThreshold: self.iouThreshold,
                            limit: 20
                        )
                    )
                }

                // Build Swift detection results directly — no UnsafePointer conversion needed
                let detections = nmsPredictions.map { pred -> YOLODetection in
                    let p1 = coordinateRestorer(pred.xyxy.x1, pred.xyxy.y1)
                    let p2 = coordinateRestorer(pred.xyxy.x2, pred.xyxy.y2)
                    // Centroid = bbox center (no OpenCV contours needed)
                    let cx = (p1.0 + p2.0) / 2.0
                    let cy = (p1.1 + p2.1) / 2.0

                    return YOLODetection(
                        classIndex: pred.classIndex,
                        className: self.classNames[pred.classIndex, default: "unknown"],
                        confidence: pred.score,
                        boxX1: Int(p1.0 * scaleX),
                        boxY1: Int(p1.1 * scaleY),
                        boxX2: Int(p2.0 * scaleX),
                        boxY2: Int(p2.1 * scaleY),
                        centroidX: Int(cx * scaleX),
                        centroidY: Int(cy * scaleY)
                    )
                }

                let result = YOLODetectionResult(detections: detections, timestamp: timestamp)
                yoloCallback?(result)
            }
        }
    }
}


class ThresholdProvider: MLFeatureProvider, @unchecked Sendable {
    var values: [String: MLFeatureValue]

    var featureNames: Set<String> {
        return Set(values.keys)
    }

    nonisolated init(iouThreshold: Double = 0.45, confidenceThreshold: Double = 0.25) {
        values = [
            "iouThreshold": MLFeatureValue(double: iouThreshold),
            "confidenceThreshold": MLFeatureValue(double: confidenceThreshold),
        ]
    }

    nonisolated func featureValue(for featureName: String) -> MLFeatureValue? {
        return values[featureName]
    }
}


class YOLORequest: VNCoreMLRequest, @unchecked Sendable {
    var userData: [String: Any] = [:]
}
