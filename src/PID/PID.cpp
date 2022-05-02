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
	proportional = gp * error;	
	differentiator = gd * (error - prevError)/time;	
	integrator = gi *(integrator + error * time);
	
	if (integrator > maxLimitI) {

		integrator = maxLimitI;
	}
	else if (integrator < minLimitI) {

		integrator = maxLimitI;
	}
	output = proportional + integrator + differentiator;
	if (output > maxOutputLimit) {
		output = maxOutputLimit;
	}
	else if (output < minOutputLimit) {
		output = minOutputLimit;
	}
	prevError = error;
	
	return -output;
}

