import CoreML

extension Collection {
    nonisolated func concurrentMap<T>(_ transform: (Element) -> T) -> [T] {
        let n = count
        if n == 0 {
            return []
        }

        // For small collections, don't parallelize
        if n < 32 {
            return map(transform)
        }

        let threadCount = Swift.min(ProcessInfo.processInfo.activeProcessorCount, n)
        let jobsPerThread = Swift.max(1, n / threadCount)

        var result = [T?](repeating: nil, count: n)
        DispatchQueue.concurrentPerform(iterations: threadCount) { thread in
            let start = thread * jobsPerThread
            let end = (thread == threadCount - 1) ? n : start + jobsPerThread
            for i in start..<end {
                let index = self.index(self.startIndex, offsetBy: i)
                result[i] = transform(self[index])
            }
        }

        return result.compactMap { $0 }
    }

    nonisolated func concurrentEnumeratedMap<T>(_ transform: (Int, Element) -> T) -> [T] {
        let n = count
        if n == 0 {
            return []
        }

        if n < 32 {
            return enumerated().map { transform($0.offset, $0.element) }
        }

        let threadCount = Swift.min(ProcessInfo.processInfo.activeProcessorCount, n)
        let jobsPerThread = Swift.max(1, n / threadCount)

        var result = [T?](repeating: nil, count: n)
        DispatchQueue.concurrentPerform(iterations: threadCount) { thread in
            let start = thread * jobsPerThread
            let end = (thread == threadCount - 1) ? n : start + jobsPerThread
            for i in start..<end {
                let index = self.index(self.startIndex, offsetBy: i)
                result[i] = transform(i, self[index])
            }
        }

        return result.compactMap { $0 }
    }
}

extension MLMultiArray {
    nonisolated func flatArray() -> [Float] {
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(self.dataPointer))
        let buffer = UnsafeBufferPointer(start: pointer, count: self.count)
        return Array(buffer)
    }
}
