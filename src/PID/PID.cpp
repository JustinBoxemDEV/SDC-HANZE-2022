#include "PID.h"
#include <iostream>
#include <cmath>

void PIDController::PIDController_Init() {
	minOutputLimit = -1.0;
	maxOutputLimit = 1.0;
	minLimitI = -1.0;
	maxLimitI = 1.0;
	time = 0.03333333333;
	integrator = 0.0;
	prevError = 0.0;
	differentiator = 0.0;
	output = 0.0;
}


double PIDController::PIDController_update(double error) {


	proportional = gp * error;	
	differentiator = gd * (error - prevError)/time;	
	integrator = gi *(integrator + error * time);
	// std::cout<<"error/time"<<error/time<<std::endl;
	// std::cout<<"P"<<proportional<<std::endl;
	// std::cout<<"I"<<integrator<<std::endl;
	// std::cout<<"D"<<differentiator<<std::endl;

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
	
	return output;
}

double PIDController::calculateTest(double pidout){

    double hoek = pidout *100.0/4.2;
    return double(sin(hoek*M_PI/180.0)*11.1111); //versnelling 1 cm/s
	//return double(sin(hoek*M_PI/180.0)*23.1481); //versnelling 2 cm/s
    //return double(sin(hoek*M_PI/180.0)*37.0370); //versnelling 3 cm/s

}