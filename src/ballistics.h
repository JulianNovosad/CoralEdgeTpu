#ifndef BALLISTICS_H
#define BALLISTICS_H

#include "pipeline_structs.h"

class BallisticsCalculator {
public:
    BallisticsCalculator();
    ~BallisticsCalculator();

    void calculate(const DetectionResult& detection, double distance);
};

#endif // BALLISTICS_H
