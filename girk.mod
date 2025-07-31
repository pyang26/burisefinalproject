NEURON {
   SUFFIX girk
   USEION k READ ek WRITE ik
   RANGE gk_girk, g, ik
}


UNITS {
   (mV) = (millivolt)
   (mA) = (milliamp)
   (S)  = (siemens)
}


PARAMETER {
   gk_girk = 0.001 (S/cm2)  : Max conductance
   vhalfl = -90  (mV)       : Half-activation voltage
   slope = 10    (mV)       : Slope factor
}


ASSIGNED {
   v     (mV)
   ek    (mV)
   g     (S/cm2)
   ik    (mA/cm2)
}


BREAKPOINT {
   g = gk_girk / (1 + exp((v - vhalfl)/slope))
   ik = g * (v - ek)
}


