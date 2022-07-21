# Machine Learning
This folder contains all of the Machine Learning code used throughout the SDC 2022.

# Folder structure
**/ReinforcementLearning:**  </br>
* **/Assetto Corsa:** Code for Reinforcement learning in Assettto Corsa </br>
* **/RLExamples** Reinforcement learning code examples </br>

**/SupervisedLearning:** Code for the SL approach: SelfDriveModel + DirectionClassificationModel  </br>
* **/Classification:** Code for training/deploying a YOLOv5 or custom-made classification model  </br>
* **/ComboNetwork:** Code for deploying a combination network SelfDriveModel + DirectionClassificationModel </br>
* **/HelperScripts** All Python scripts that have been used throughout the project for creating/modifying the dataset etc</br>
* **/tensorboards** (git ignored due to size) All of the saved tensorboard sessions </br>

# Additional Info
To train a SelfDriveModel use the **/SupervisedLearning/SL_self_drive_pipeline.py**. </br>
To train a DirectionClassificationModel use **/SupervisedLearning/Classification/my_classifier.py**.