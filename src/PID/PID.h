#ifndef PID_H
#define PID_H

class PIDController {
	private:
		//Gains (waardes moeten nog worden ingesteld)
		double gp;
		double gi;
		double gd;
		//waardes moeten nog gevonden worden
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
	PIDController(double gp, double gi, double gd){
		this->gp = gp;
		this->gi = gi;
		this->gd = gd;
	}
	void PIDController_Init();
	double PIDController_update( double error);
	double calculateTest(double pidout);
};
#endif 
