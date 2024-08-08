const { ModelChecker } = require('./model_checker');
const { TheoremProver } = require('./theorem_prover');

class FormalVerifier {
    async verifyContract(contractAddress, data) {
        const modelChecker = new ModelChecker();
        const theoremProver = new TheoremProver();

        const modelCheckingResult = await modelChecker.verify(contractAddress, data);
        const theoremProvingResult = await theoremProver.verify(contractAddress, data);

        return { modelCheckingResult, theoremProvingResult };
    }
}

module.exports = FormalVerifier;
