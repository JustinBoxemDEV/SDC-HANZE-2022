#include "VehicleControlManager.h"

VehicleControlManager::VehicleControlManager(CommunicationStrategy *vehicleStrategy__) {
    vehicleStrategy = vehicleStrategy__;
};

void VehicleControlManager::throttle() {
    vehicleStrategy->throttle();
};

void VehicleControlManager::steer() {
    vehicleStrategy->steer();
};

void VehicleControlManager::brake() {
    vehicleStrategy->brake();
};

void VehicleControlManager::stop() {
    vehicleStrategy->stop();
};

void VehicleControlManager::neutral() {
    vehicleStrategy->neutral();
};