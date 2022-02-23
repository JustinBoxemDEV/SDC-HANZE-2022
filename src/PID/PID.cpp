#include "PID.h";
#include <iostream>

void PIDController::PIDController_Init(PIDController) {
	gp = 0.2;
	gi = 0.05;
	gd = 0.025;
	
	lowPassFitler = 0.02;
	minOutputLimit = -1;
	maxOutputLimit = 1;
	minLimitI = -1;
	maxLimitI = 1;
	time = 0.03333333333; // bereken met 1/fps
	integrator = 0;
	prevError = 0;
	differentiator = 0;
	output = 0;
}


double PIDController::PIDController_update(PIDController, double error) {

	
	proportional = gp * error;
	std::cout << "prop " << proportional << std::endl;// voor debugging

	if ((error < 0 && prevError > 0) || (error > 0 && prevError < 0) || error == 0) {
		time = 0.03333333333;// time = 1/fps
	}
	//calculate I and clamp
	integrator = integrator + 0.5 * gi * time * (error + prevError);

	if (integrator > maxLimitI) {

		integrator = maxLimitI;
	}
	else if (integrator < minLimitI) {

		integrator = maxLimitI;
	}
	std::cout << "inte " << integrator << std::endl;// voor debugging
	
	differentiator = -(2.0 * gd * (error - prevError) + (2.0 * lowPassFitler - time) * differentiator) / (2.0 * lowPassFitler + time);//<== verander
	std::cout << "diff " << differentiator << std::endl;//voor debugging
	//calculate output and clamp
	output = proportional + integrator + differentiator;
	if (output > maxOutputLimit) {
		output = maxOutputLimit;
	}
	else if (output < minOutputLimit) {
		output = minOutputLimit;
	}
	time = time + 0.03333333333; // time + 1/fps

	prevError = error;

	return output;
}

