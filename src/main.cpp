#include "Managers/VehicleControlManager.h"
#include "VehicleControl/strategies/CANStrategy.h"

int main() {
    CANStrategy *strat = new CANStrategy();
    VehicleControlManager vehicleControlManager(strat);

    vehicleControlManager.forward(20);
};