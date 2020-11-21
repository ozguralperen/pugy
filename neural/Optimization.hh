#ifndef OPT
#define OPT
#include <cmath>
#include <algorithm>

#define LEARNINGRATE  0.001
#define WEIGHT 0
double opt1 = 0.9 , opt2 = 0.999;
int epoch = 1;

void UpdateWeights(double grad , double& m , double& v , double& dw3 , double w){
	w -= LEARNINGRATE * WEIGHT * w;

    m = opt1 * m + (1 - opt1) * grad;
	v = opt2 * v + (1 - opt2) * grad * grad;

	double mt = m / (1 - pow(opt1, epoch));
	double vt = v / (1 - pow(opt1, epoch));

	w -= LEARNINGRATE * mt / (sqrt(vt) + 1e-8);
}

#endif