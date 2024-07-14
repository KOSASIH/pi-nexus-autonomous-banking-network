// File name: quantum_optimization.qs
namespace QuantumOptimization {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;

    operation OptimizePortfolio(assets : Double[]) : Result {
        using (qubits = Qubit[assets.Length]) {
            ApplyToEach(H, qubits);
            for (i in 0 .. assets.Length - 1) {
                ApplyIf(assets[i] > 0.5, X, qubits[i]);
            }
            ApplyToEach(M, qubits);
            let result = Measure(qubits);
            Reset(qubits);
            return result;
        }
    }
}
