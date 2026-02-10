import Foundation

enum ParticleEffectType: String, CaseIterable {
    case fire
    case smoke
    case sparks
    case rain
    case snow
    case magic
    case impact
    case debugCube

    var displayName: String {
        rawValue.capitalized
    }

    var icon: String {
        switch self {
        case .fire: return "flame.fill"
        case .smoke: return "cloud.fill"
        case .sparks: return "sparkles"
        case .rain: return "cloud.rain.fill"
        case .snow: return "snowflake"
        case .magic: return "wand.and.stars"
        case .impact: return "burst.fill"
        case .debugCube: return "cube.fill"
        }
    }
}
