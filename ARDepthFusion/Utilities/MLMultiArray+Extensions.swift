import CoreML

extension MLMultiArray {
    nonisolated func floatValue(at indices: [Int]) -> Float? {
        for (dim, idx) in indices.enumerated() {
            guard dim < shape.count, idx >= 0, idx < shape[dim].intValue else {
                return nil
            }
        }
        let nsIndices = indices.map { NSNumber(value: $0) }
        return self[nsIndices].floatValue
    }

    nonisolated var totalCount: Int {
        shape.reduce(1) { $0 * $1.intValue }
    }

    nonisolated func toFloatArray() -> [Float] {
        let count = totalCount
        let ptr = dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: count))
    }
}
