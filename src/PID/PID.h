#ifndef PID_H
#define PID_H

class PIDController {
	//Gains (waardes moeten nog worden ingesteld)
	double gp;
	double gi;
	double gd;
	//waardes moeten nog gevonden worden
	double lowPassFitler;
	double minOutputLimit;//-1
	double maxOutputLimit;//1
	double minLimitI;
	double maxLimitI;
	// tijd is seconden
	double time;

	double proportional;
	double integrator;
	double differentiator;
	double prevError;
	double prevMesurement;

	double output;


public:
	void PIDController_Init(PIDController);
	double PIDController_update(PIDController pid, double setpoint, double mesurment);
};
#endif 
