import Foundation

enum EffectType: String, CaseIterable, Identifiable {
    case explosion
    case fireball
    case flamethrower
    case smoke
    case lightning
    case magic
    case debugCube

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .fireball: return "Fireball"
        case .flamethrower: return "Flamethrower"
        case .debugCube: return "Debug Cube"
        default: return rawValue.capitalized
        }
    }

    var icon: String {
        switch self {
        case .explosion: return "burst.fill"
        case .fireball: return "flame.fill"
        case .flamethrower: return "flame"
        case .smoke: return "cloud.fill"
        case .lightning: return "bolt.fill"
        case .magic: return "wand.and.stars"
        case .debugCube: return "cube.fill"
        }
    }

    /// Returns the video file name (without extension) for this effect, or nil for debugCube.
    var videoFileName: String? {
        switch self {
        case .debugCube: return nil
        default: return rawValue
        }
    }

    /// Whether this effect's resources are available in the bundle.
    var isAvailable: Bool {
        if self == .debugCube { return true }
        guard let fileName = videoFileName else { return false }
        return Bundle.main.url(forResource: fileName, withExtension: "mov") != nil
    }

    /// All effect types whose resources exist in the bundle.
    static let available: [EffectType] = allCases.filter { $0.isAvailable }
}
