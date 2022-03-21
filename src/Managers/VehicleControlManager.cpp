#include "VehicleControlManager.h"

VehicleControlManager::VehicleControlManager(CommunicationStrategy *vehicleStrategy__) {
    vehicleStrategy = vehicleStrategy__;
};

void VehicleControlManager::forward(int amount) {
    vehicleStrategy->forward(amount);
};

void VehicleControlManager::steer(float amount) {
    vehicleStrategy->steer(amount);
};

void VehicleControlManager::brake(int amount) {
    vehicleStrategy->brake(amount);
};

void VehicleControlManager::stop() {
    vehicleStrategy->stop();
};

void VehicleControlManager::neutral() {
    vehicleStrategy->neutral();
};