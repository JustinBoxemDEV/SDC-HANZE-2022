#include PID.h

void PIDController_Init(PIDController pid) {

	integrator = 0;
	prevError = 0;
	differentiator = 0;
	prevMesurement = 0;
	output = 0;
}

double PIDController_Update(PIDController pid, double setpoint, double mesurment) {

	double error = setpoint - mesurement;

	proportional = pid->gp * error;


	//calculate I and clamp
	pid->intergrator = pid->intergrator + 0.5 * pid->gi * pid->time * (error + pid->prevError);
	if (pid->integrator > pid->limMaxInt) {

		pid->integrator = pid->limMaxInt;
	}
	else if (pid->integrator < pid->limMinInt) {

		pid->integrator = pid->limMinInt;
	}

	
	pid->differentiator = -(2.0 * pid->gd * (measurement - pid->prevMeasurement) + (2.0 * pid->lowPassFitler - pid->time) * pid->differentiator) / (2.0 * pid->lowPassFitler + pid->time);

	//calculate output and clamp
	pid->output = proportinal + pid->intergrator + pid->differentiator;
	if (pid->output > pid->maxOutputLimit) {
		pid->output = pid->maxOutputLimit;
	}
	else if (pid->output < minOutputLimit) {
		pid->output = minOutputLimit;
	}


	pid->prevError = error;
	pid->prevMeasurement = measurement;

	return output
}