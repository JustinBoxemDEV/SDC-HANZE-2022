#include "vehiclecontrolmanager.h"

VehicleControlManager::VehicleControlManager(CommunicationStrategy *vehicleStrategy__) {
    vehicleStrategy = vehicleStrategy__;
};

void VehicleControlManager::forward() {
    vehicleStrategy->forward();
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