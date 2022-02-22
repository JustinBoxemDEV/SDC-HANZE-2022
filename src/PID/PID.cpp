#include "PID.h";

void PIDController_Init(PIDController pid) {

	pid.integrator = 0;
	pid.prevError = 0;
	pid.differentiator = 0;
	pid.prevMesurement = 0;
	pid.output = 0;
}

double PIDController_Update(PIDController pid, double setpoint, double mesurment) {

	double error = setpoint - mesurment;

	pid.proportional = pid.gp * error;


	//calculate I and clamp
	pid.integrator = pid.integrator + 0.5 * pid.gi * pid.time * (error + pid.prevError);
	if (pid.integrator > pid.maxLimitI) {

		pid.integrator = pid.maxLimitI;
	}
	else if (pid.integrator < pid.minLimitI) {

		pid.integrator = pid.maxLimitI;
	}

	
	pid.differentiator = -(2.0 * pid.gd * (mesurment - pid.prevMesurement) + (2.0 * pid.lowPassFitler - pid.time) * pid.differentiator) / (2.0 * pid.lowPassFitler + pid.time);

	//calculate output and clamp
	pid.output = pid.proportional + pid.integrator + pid.differentiator;
	if (pid.output > pid.maxOutputLimit) {
		pid.output = pid.maxOutputLimit;
	}
	else if (pid.output < pid.minOutputLimit) {
		pid.output = pid.minOutputLimit;
	}


	pid.prevError = error;
	pid.prevMesurement = mesurment;

	return pid.output;
}