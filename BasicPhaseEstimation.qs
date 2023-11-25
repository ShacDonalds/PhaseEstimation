namespace PhaseEstimation {

    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Math;

    operation U (phi : Double, psi : Qubit[]): Unit is Adj + Ctl {
        Controlled Rz (Most(psi), (2.*phi, Tail(psi)));    
    }
    
    operation basicPhaseEstimation (n : Int, theta : Double, oracle : (Qubit[]) => Unit is Ctl, psi : Qubit[]) : Result {

        use aux = Qubit();

        H(aux);
        Rz (-IntAsDouble(n)*theta, aux);
        for _ in 0 .. n-1 {
            Controlled oracle ([aux], psi);
        }
        H(aux);
        let result = MResetZ(aux);

        return result;
    }

    // @EntryPoint()
    operation run (nShots : Int, phi : Double, oraclePower : Int) : (Int, Int) {

        let theta = - PI() / (2.0 * IntAsDouble(oraclePower));
        use psi = Qubit[3];
        ApplyToEachCA(X, psi);

        mutable nZero = 0;
        mutable nOne = 0; 
        for _ in 0 .. nShots-1 {
            let result = basicPhaseEstimation(oraclePower, theta, U(phi, _), psi);
            if (result == One) {
                set nOne += 1;
            } else {
                set nZero += 1;
            }
        }
        
        ResetAll(psi);
        return (nZero, nOne);
    }

}
