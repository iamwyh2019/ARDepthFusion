import Foundation

struct DetectedObject: Identifiable {
    let id = UUID()
    let className: String
    let confidence: Float
    let boundingBox: CGRect
    let centroid: CGPoint
    let mask: [Float]?
    let maskWidth: Int
    let maskHeight: Int
}
