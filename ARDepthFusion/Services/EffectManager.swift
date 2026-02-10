import RealityKit
import Observation

@Observable
final class EffectManager {
    var placedEffects: [PlacedEffect] = []

    func placeEffect(
        type: ParticleEffectType,
        objectClass: String,
        at position: SIMD3<Float>,
        scale: Float,
        in arView: ARView
    ) {
        let anchor = AnchorEntity(world: position)
        let entity = ModelEntity()

        var emitter = configureEmitter(for: type, scale: scale)
        entity.components.set(emitter)
        anchor.addChild(entity)
        arView.scene.addAnchor(anchor)

        let effect = PlacedEffect(
            type: type,
            objectClass: objectClass,
            anchor: anchor
        )
        placedEffects.append(effect)
    }

    func removeEffect(_ effect: PlacedEffect, from arView: ARView) {
        effect.anchor.removeFromParent()
        placedEffects.removeAll { $0.id == effect.id }
    }

    func clearAll(from arView: ARView) {
        for effect in placedEffects {
            effect.anchor.removeFromParent()
        }
        placedEffects.removeAll()
    }

    private func configureEmitter(
        for type: ParticleEffectType,
        scale: Float
    ) -> ParticleEmitterComponent {
        var emitter = ParticleEmitterComponent()
        let s = scale

        switch type {
        case .fire:
            emitter.emitterShape = .cone
            emitter.emitterShapeSize = SIMD3<Float>(s * 0.3, s * 0.5, s * 0.3)
            emitter.birthRate = 200
            emitter.mainEmitter.lifeSpan = 0.8
            emitter.speed = 0.3
            emitter.mainEmitter.size = 0.03 * s
            emitter.mainEmitter.color = .evolving(
                start: .single(.init(red: 1.0, green: 0.9, blue: 0.2, alpha: 1.0)),
                end: .single(.init(red: 1.0, green: 0.2, blue: 0.0, alpha: 0.0))
            )
            emitter.mainEmitter.birthDirection = .world
            emitter.birthDirection = .normal

        case .smoke:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = SIMD3<Float>(s * 0.4, s * 0.4, s * 0.4)
            emitter.birthRate = 80
            emitter.mainEmitter.lifeSpan = 2.0
            emitter.speed = 0.1
            emitter.mainEmitter.size = 0.05 * s
            emitter.mainEmitter.color = .evolving(
                start: .single(.init(red: 0.5, green: 0.5, blue: 0.5, alpha: 0.6)),
                end: .single(.init(red: 0.7, green: 0.7, blue: 0.7, alpha: 0.0))
            )

        case .sparks:
            emitter.emitterShape = .point
            emitter.birthRate = 150
            emitter.mainEmitter.lifeSpan = 0.5
            emitter.speed = 2.0
            emitter.mainEmitter.size = 0.01 * s
            emitter.mainEmitter.color = .evolving(
                start: .single(.init(red: 1.0, green: 1.0, blue: 0.8, alpha: 1.0)),
                end: .single(.init(red: 1.0, green: 0.6, blue: 0.0, alpha: 0.0))
            )

        case .rain:
            emitter.emitterShape = .plane
            emitter.emitterShapeSize = SIMD3<Float>(s * 1.0, 0, s * 1.0)
            emitter.birthRate = 300
            emitter.mainEmitter.lifeSpan = 1.5
            emitter.speed = 3.0
            emitter.mainEmitter.size = 0.005 * s
            emitter.mainEmitter.color = .constant(.single(.init(
                red: 0.7, green: 0.85, blue: 1.0, alpha: 0.6
            )))
            emitter.birthDirection = .constant
            emitter.emissionDirection = SIMD3<Float>(0, -1, 0)

        case .snow:
            emitter.emitterShape = .plane
            emitter.emitterShapeSize = SIMD3<Float>(s * 1.0, 0, s * 1.0)
            emitter.birthRate = 100
            emitter.mainEmitter.lifeSpan = 3.0
            emitter.speed = 0.3
            emitter.mainEmitter.size = 0.015 * s
            emitter.mainEmitter.color = .constant(.single(.init(
                red: 1.0, green: 1.0, blue: 1.0, alpha: 0.9
            )))
            emitter.birthDirection = .constant
            emitter.emissionDirection = SIMD3<Float>(0, -1, 0)

        case .magic:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = SIMD3<Float>(s * 0.5, s * 0.5, s * 0.5)
            emitter.birthRate = 50
            emitter.mainEmitter.lifeSpan = 1.5
            emitter.speed = 0.5
            emitter.mainEmitter.size = 0.02 * s
            emitter.mainEmitter.color = .evolving(
                start: .single(.init(red: 0.6, green: 0.2, blue: 1.0, alpha: 1.0)),
                end: .single(.init(red: 0.0, green: 0.8, blue: 1.0, alpha: 0.0))
            )

        case .impact:
            emitter.emitterShape = .sphere
            emitter.emitterShapeSize = SIMD3<Float>(s * 0.1, s * 0.1, s * 0.1)
            emitter.birthRate = 500
            emitter.mainEmitter.lifeSpan = 0.4
            emitter.speed = 3.0
            emitter.mainEmitter.size = 0.015 * s
            emitter.mainEmitter.color = .evolving(
                start: .single(.init(red: 1.0, green: 0.5, blue: 0.0, alpha: 1.0)),
                end: .single(.init(red: 1.0, green: 0.1, blue: 0.0, alpha: 0.0))
            )
        }

        return emitter
    }
}
