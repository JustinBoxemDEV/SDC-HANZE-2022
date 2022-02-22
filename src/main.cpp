#include <iostream>
#include "logger/logger.h"

int main() {
    Logger logger;

    logger.debug("Debug log");
    logger.error("Error log");
    logger.info("Info log");
    logger.success("Success log");
    logger.log("Default log");

    logger.createFile();
    logger.debug("Debug log");
    logger.error("Error log");
    logger.info("Info log");
    logger.success("Success log");
    logger.log("Default log");
    
    logger.createFile("Laurens");
    logger.log("125 kilo");
}