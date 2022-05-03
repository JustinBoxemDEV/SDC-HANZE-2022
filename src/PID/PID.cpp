#include "PID.h"
#include <iostream>

void PIDController::PIDController_Init() {
	minOutputLimit = -1;
	maxOutputLimit = 1;
	time = 0.03333333333;
	integrator = 0;
	prevError = 0;
	differentiator = 0;
	output = 0;
}

double PIDController::PIDController_update(double error) {
    proportional = error;    
    differentiator = (error - prevError) /time;    
    integrator = integrator + error * time;
    
    output = (gp * proportional) + (gi * integrator) + (gd * differentiator);
    if (output > maxOutputLimit) {
        output = maxOutputLimit;
    }
    else if (output < minOutputLimit) {
        output = minOutputLimit;
    }
    prevError = error;
    std::cout<<"proportional= " << proportional <<std::endl;
    std::cout<<"intergator= " << integrator<<std::endl;
    std::cout<<"differentiator= " << differentiator<<std::endl;
    return output;
}

