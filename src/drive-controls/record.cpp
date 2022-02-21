#include "record.h"
#include <iostream>

std::map<short, std::string> Record::CanListener::idConversion {
    std::make_pair(0x12c, "steering"),
    std::make_pair(0x120, "throttle"),
    std::make_pair(0x126, "brake")
};

Record::CanListener::CanListener() {
    
};

void Record::CanListener::startListenting() {

};

void Record::CanListener::stopListening() {

};

void Record::CanListener::getNewValues() {

};

void Record::CanListener::listen() {

};

Record::ImageWorker::ImageWorker(std::queue<int> imageQueue) {

};

void Record::ImageWorker::start() {

};

void Record::ImageWorker::stop() {

};

void Record::ImageWorker::put() {

};

void Record::ImageWorker::process() {

};

Record::CanWorker::CanWorker(std::queue<int> canQueue, char* folderName) {

};

void Record::CanWorker::start() {

};

void Record::CanWorker::stop() {

};

void Record::CanWorker::put() {

};

void Record::CanWorker::process() {

};

// represents the main function in the python file
void Record::run() {

};
