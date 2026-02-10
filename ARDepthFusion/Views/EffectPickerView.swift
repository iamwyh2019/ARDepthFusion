import SwiftUI

struct EffectPickerView: View {
    let objectClass: String
    let onSelect: (ParticleEffectType) -> Void

    var body: some View {
        VStack(spacing: 16) {
            Text("Add Effect to \(objectClass)")
                .font(.headline)

            LazyVGrid(columns: [
                GridItem(.adaptive(minimum: 70), spacing: 12)
            ], spacing: 12) {
                ForEach(ParticleEffectType.allCases, id: \.self) { type in
                    Button {
                        onSelect(type)
                    } label: {
                        VStack(spacing: 6) {
                            Image(systemName: type.icon)
                                .font(.title2)
                                .frame(width: 44, height: 44)
                                .background(Color.accentColor.opacity(0.15))
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                            Text(type.displayName)
                                .font(.caption2)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding()
    }
}
