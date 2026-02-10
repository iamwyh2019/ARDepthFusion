import CoreML
import Vision
import UIKit
import Accelerate

typealias XYXY = (x1: Float, y1: Float, x2: Float, y2: Float)

struct BoxPrediction: Sendable {
    let classIndex: Int
    let score: Float
    let xyxy: XYXY
    let maskCoefficients: [Float]
}

nonisolated func parseBoundingBoxes(
    multiArray: MLMultiArray,
    numClasses: Int,
    confidenceThreshold: Float
) -> [BoxPrediction] {
    let boxCount = multiArray.shape[2].intValue
    let featureCount = multiArray.shape[1].intValue
    let numMasks = featureCount - (4 + numClasses)
    let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
    let stride = boxCount

    let bboxStart    = pointer
    let classStart   = pointer.advanced(by: 4 * stride)
    let maskStartAll = pointer.advanced(by: (4 + numClasses) * stride)

    return (0..<boxCount).concurrentMap { i -> BoxPrediction? in
        var maxConfidence: Float = 0
        var bestClassIndex = -1

        let thisClassStart = classStart.advanced(by: i)
        vDSP_maxvi(thisClassStart, stride, &maxConfidence, &bestClassIndex, vDSP_Length(numClasses))
        guard maxConfidence > confidenceThreshold, bestClassIndex >= 0 else { return nil }
        bestClassIndex /= stride

        let cx = bboxStart.advanced(by: 0 * stride)[i]
        let cy = bboxStart.advanced(by: 1 * stride)[i]
        let w  = bboxStart.advanced(by: 2 * stride)[i]
        let h  = bboxStart.advanced(by: 3 * stride)[i]

        let maskCoefficients = (0..<numMasks).map { m -> Float in
            maskStartAll.advanced(by: m * stride)[i]
        }

        return BoxPrediction(
            classIndex: bestClassIndex,
            score: maxConfidence,
            xyxy: XYXY(
                x1: cx - w / 2,
                y1: cy - h / 2,
                x2: cx + w / 2,
                y2: cy + h / 2
            ),
            maskCoefficients: maskCoefficients
        )
    }
    .compactMap { $0 }
}


nonisolated func nonMaximumSuppression(
    predictions: [BoxPrediction],
    iouThreshold: Float,
    limit: Int
) -> [BoxPrediction] {
    guard !predictions.isEmpty else { return [] }

    let sortedIndices = predictions.indices.sorted {
        predictions[$0].score > predictions[$1].score
    }

    var selected: [BoxPrediction] = []
    var active = [Bool](repeating: true, count: predictions.count)
    var numActive = active.count

    outer: for i in 0..<predictions.count {
        if active[i] {
            let boxA = predictions[sortedIndices[i]]
            selected.append(boxA)

            if selected.count >= limit { break }

            for j in i+1..<predictions.count {
                if active[j] {
                    let boxB = predictions[sortedIndices[j]]

                    if IOU(box1: boxA.xyxy, box2: boxB.xyxy) > iouThreshold {
                        active[j] = false
                        numActive -= 1

                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
    return selected
}


nonisolated func getMaskProtos(masks: MLMultiArray, numMasks: Int) -> [[Float]] {
    let width = masks.shape[2].intValue
    let height = masks.shape[3].intValue
    let maskSize = height * width

    let pointer = masks.dataPointer.assumingMemoryBound(to: Float.self)
    let maskStride = masks.strides[1].intValue

    return (0..<numMasks).concurrentMap { maskIdx -> [Float] in
        let start = pointer.advanced(by: maskIdx * maskStride)
        return Array(UnsafeBufferPointer(start: start, count: maskSize))
    }
}


nonisolated func getMasksFromProtos(maskProtos: [[Float]], coefficients: [Float]) -> [Float] {
    guard !maskProtos.isEmpty, maskProtos.count == coefficients.count else { return [] }

    let maskSize = maskProtos[0].count
    var result = [Float](repeating: 0, count: maskSize)

    for (proto, coefficient) in zip(maskProtos, coefficients) {
        vDSP_vsma(proto, 1, [coefficient], result, 1, &result, 1, vDSP_Length(maskSize))
    }

    return result
}


nonisolated func getSigmoidMask(mask: UnsafePointer<Float>, maskSize: Int) -> [Float] {
    var count = Int32(maskSize)
    var onef: Float = 1.0
    var negOne: Float = -1.0

    var expNegatedMask = [Float](repeating: 0, count: maskSize)
    vDSP_vsmul(mask, 1, &negOne, &expNegatedMask, 1, vDSP_Length(maskSize))
    vvexpf(&expNegatedMask, expNegatedMask, &count)

    vDSP_vsadd(expNegatedMask, 1, &onef, &expNegatedMask, 1, vDSP_Length(maskSize))

    var sigmoidMask = [Float](repeating: 0, count: maskSize)
    vvrecf(&sigmoidMask, &expNegatedMask, &count)

    return sigmoidMask
}


nonisolated func cropMask(mask: [Float], width: Int, height: Int, bbox: XYXY) -> [Float] {
    var result = [Float](repeating: 0, count: width * height)
    let x1 = max(0, Int(bbox.x1))
    let y1 = max(0, Int(bbox.y1))
    let x2 = min(width - 1, Int(bbox.x2))
    let y2 = min(height - 1, Int(bbox.y2))
    let rowWidth = x2 - x1 + 1

    DispatchQueue.concurrentPerform(iterations: y2 - y1 + 1) { i in
       let y = y1 + i
       let offset = y * width + x1
       mask.withUnsafeBufferPointer { ptr in
           vDSP_vsmul(ptr.baseAddress! + offset, 1, [1.0], &result[offset], 1, vDSP_Length(rowWidth))
       }
    }

    return result
}


nonisolated func cropMaskPhysical(mask: [Float], width: Int, height: Int, bbox: XYXY) -> ([Float], (width: Int, height: Int)) {
   let x1 = max(0, Int(bbox.x1))
   let y1 = max(0, Int(bbox.y1))
   let x2 = min(width - 1, Int(bbox.x2))
   let y2 = min(height - 1, Int(bbox.y2))
   let cropWidth = x2 - x1 + 1
   let cropHeight = y2 - y1 + 1

   var cropped = [Float](repeating: 0, count: cropWidth * cropHeight)

   mask.withUnsafeBufferPointer { srcPtr in
       cropped.withUnsafeMutableBufferPointer { dstPtr in
           DispatchQueue.concurrentPerform(iterations: cropHeight) { y in
               let srcOffset = (y + y1) * width + x1
               let dstOffset = y * cropWidth
               memcpy(dstPtr.baseAddress! + dstOffset,
                     srcPtr.baseAddress! + srcOffset,
                     cropWidth * MemoryLayout<Float>.stride)
           }
       }
   }

   return (cropped, (cropWidth, cropHeight))
}


nonisolated func upsampleMask(mask: UnsafePointer<Float>, width: Int, height: Int, newWidth: Int, newHeight: Int) -> [Float] {
   let sourceRowBytes = width * MemoryLayout<Float>.stride

   var sourceBuffer = vImage_Buffer(
       data: UnsafeMutableRawPointer(mutating: mask),
       height: vImagePixelCount(height),
       width: vImagePixelCount(width),
       rowBytes: sourceRowBytes
   )

   var destinationBuffer = try! vImage_Buffer(
       width: Int(newWidth),
       height: Int(newHeight),
       bitsPerPixel: 32
   )

   let error = vImageScale_PlanarF(
       &sourceBuffer,
       &destinationBuffer,
       nil,
       vImage_Flags(kvImageNoFlags)
   )

   guard error == kvImageNoError else {
       destinationBuffer.free()
       NSLog("Error during upsampling: \(error)")
       return []
   }

   let result = Array(
       UnsafeBufferPointer(
           start: destinationBuffer.data.assumingMemoryBound(to: Float.self),
           count: newWidth * newHeight
       )
   )

   destinationBuffer.free()
   return result
}


nonisolated func removeBelowThreshold(mask: [Float], threshold: Float = 0.5) -> [Float] {
    return vDSP.threshold(mask, to: threshold, with: .zeroFill)
}


nonisolated func sigmoid(value: Float) -> Float {
    return 1.0 / (1.0 + exp(-value))
}

nonisolated func IOU(box1: XYXY, box2: XYXY) -> Float {
    let xA = max(box1.x1, box2.x1)
    let yA = max(box1.y1, box2.y1)
    let xB = min(box1.x2, box2.x2)
    let yB = min(box1.y2, box2.y2)

    if xA >= xB || yA >= yB {
        return 0.0
    }

    let interArea = (xB - xA) * (yB - yA)
    let box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    let box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

    let unionArea = box1Area + box2Area - interArea

    return interArea / unionArea
}


nonisolated func parseModelSizeAndNames(model: MLModel) -> (width: Int, height: Int, names: [Int: String]) {
    var classNames: [Int: String] = [:]
    var width: Int = -1
    var height: Int = -1

    if let metadata = model.modelDescription.metadata[.creatorDefinedKey] as? [String: Any] {
        if let namesData = metadata["names"] as? String {
            var fixedJSONString = namesData.replacingOccurrences(of: "'", with: "\"")

            if let regex = try? NSRegularExpression(pattern: "(\\d+):", options: []) {
                let range = NSRange(location: 0, length: fixedJSONString.utf16.count)
                fixedJSONString = regex.stringByReplacingMatches(in: fixedJSONString, options: [], range: range, withTemplate: "\"$1\":")
            }

            if let jsonData = fixedJSONString.data(using: .utf8) {
                do {
                    classNames = try JSONDecoder().decode([Int: String].self, from: jsonData)
                } catch {
                    NSLog("Error decoding JSON: \(error)")
                }
            } else {
                NSLog("Error: Could not convert string to data.")
            }
        } else {
            NSLog("Error: `names` field not found or invalid.")
        }

        if let sizeData = metadata["imgsz"] as? String {
            let pattern = #"^\[\s*(\d+),\s*(\d+)\s*\]$"#
            if let regex = try? NSRegularExpression(pattern: pattern) {
                if let match = regex.firstMatch(in: sizeData, range: NSRange(sizeData.startIndex..., in: sizeData)) {
                    if let range1 = Range(match.range(at: 1), in: sizeData),
                       let range2 = Range(match.range(at: 2), in: sizeData) {
                        let number1 = Int(sizeData[range1])
                        let number2 = Int(sizeData[range2])

                        if let number1 = number1, let number2 = number2 {
                            width = number1
                            height = number2
                        } else {
                            NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                        }
                    } else {
                        NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                    }
                }
                else {
                    NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
                }
            } else {
                NSLog("Error: failed to parse `imgsz` field as integers: \"\(sizeData)\".")
            }
        } else {
            NSLog("Error: `imgsz` field not found or invalid.")
        }
    } else {
        NSLog("Cannot find metadata in model description.")
    }

    return (width: width, height: height, names: classNames)
}


nonisolated func getCoordinateRestorer(
    originalSize: (width: Float, height: Float),
    targetSize: (width: Float, height: Float),
    option: VNImageCropAndScaleOption
) -> (Float, Float) -> (Float, Float) {

    let origW = originalSize.width
    let origH = originalSize.height
    let tW = targetSize.width
    let tH = targetSize.height

    var sx: Float = 1
    var sy: Float = 1
    var dx: Float = 0
    var dy: Float = 0

    switch option {
    case .scaleFit:
        let rW = tW / origW
        let rH = tH / origH
        let s = min(rW, rH)

        sx = 1.0 / s
        sy = 1.0 / s

        let scaledW = origW * s
        let scaledH = origH * s
        let padX = (tW - scaledW) / 2.0
        let padY = (tH - scaledH) / 2.0

        dx = -padX
        dy = -padY

    case .scaleFill:
        let rW = tW / origW
        let rH = tH / origH
        sx = 1.0 / rW
        sy = 1.0 / rH

    case .centerCrop:
        let rW = tW / origW
        let rH = tH / origH
        let s = max(rW, rH)

        sx = 1.0 / s
        sy = 1.0 / s

        let newW = origW * s
        let newH = origH * s
        let cropX = (newW - tW) / 2.0
        let cropY = (newH - tH) / 2.0

        dx = -cropX
        dy = -cropY

    @unknown default:
        NSLog("Seriously? What mode could it be?")
        break
    }

    return { (x: Float, y: Float) -> (Float, Float) in
        let tx = (x + dx) * sx
        let ty = (y + dy) * sy
        return (tx, ty)
    }
}
