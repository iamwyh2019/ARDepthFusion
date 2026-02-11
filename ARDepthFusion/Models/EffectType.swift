import Foundation

enum EffectType: String, CaseIterable, Identifiable {
    case flamethrower
    case explosion
    case lightning
    case dragonBreath
    case smoke
    case debugCube

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .dragonBreath: return "Dragon Breath"
        case .debugCube: return "Debug Cube"
        default: return rawValue.capitalized
        }
    }

    var icon: String {
        switch self {
        case .flamethrower: return "flame.fill"
        case .explosion: return "burst.fill"
        case .lightning: return "bolt.fill"
        case .dragonBreath: return "flame"
        case .smoke: return "cloud.fill"
        case .debugCube: return "cube.fill"
        }
    }

    /// Returns the USDZ file name (without extension) for this effect, or nil for debugCube.
    var usdzFileName: String? {
        switch self {
        case .debugCube: return nil
        case .flamethrower: return "Flamethrower"
        case .explosion: return "Explosion"
        case .lightning: return "Lightning"
        case .dragonBreath: return "DragonBreath"
        case .smoke: return "Smoke"
        }
    }

    /// Whether this effect's resources are available in the bundle.
    var isAvailable: Bool {
        if self == .debugCube { return true }
        guard let fileName = usdzFileName else { return false }
        return Bundle.main.url(forResource: fileName, withExtension: "usdz") != nil
    }

    /// All effect types whose resources exist in the bundle.
    static let available: [EffectType] = allCases.filter { $0.isAvailable }
}
