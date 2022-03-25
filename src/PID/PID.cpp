#include "PID.h"
#include <iostream>

void PIDController::PIDController_Init() {
	minOutputLimit = -1;
	maxOutputLimit = 1;
	minLimitI = -1;
	maxLimitI = 1;
	time = 0.03333333333;
	integrator = 0;
	prevError = 0;
	differentiator = 0;
	output = 0;
}


double PIDController::PIDController_update(double error) {

	if ((error < 0.0 && prevError > 0.0) || (error > 0.0 && prevError < 0.0) || (error == 0.0)) {
		time = 0.03333333333;
	}
	proportional = gp * error;	
	differentiator = gd * (error - prevError)/0.03333333333;	
	integrator = integrator + 0.5 * gi * time * (error + prevError);
	output = proportional + integrator - differentiator;

	if (integrator > maxLimitI) {

		integrator = maxLimitI;
	}
	else if (integrator < minLimitI) {

		integrator = maxLimitI;
	}

	if (output > maxOutputLimit) {
		output = maxOutputLimit;
	}
	else if (output < minOutputLimit) {
		output = minOutputLimit;
	}
	time = time + 0.03333333333;
	prevError = error;
	return output;
}

