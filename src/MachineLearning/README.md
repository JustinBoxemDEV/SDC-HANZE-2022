# Machine Learning
This folder contains all of the Machine Learning code used throughout the SDC 2022.
Author: Sabine Schreuder

# Folder structure
**/ReinforcementLearning:**  </br>
* **/Assetto Corsa:** Code for Reinforcement learning in Assettto Corsa </br>
* **/RLExamples:** Reinforcement learning code examples </br>

**/SupervisedLearning:** Code for the SL approach: SelfDriveModel + DirectionClassificationModel  </br>
* **/Classification:** Code for training/deploying a YOLOv5 or custom-made classification model  </br>
* **/ComboNetwork:** Code for deploying a combination network SelfDriveModel + DirectionClassificationModel </br>
* **/HelperScripts:** All Python scripts that have been used throughout the project for creating/modifying the dataset etc</br>
* **/tensorboards:** (git ignored due to size) All of the saved tensorboard sessions </br>

# Additional Info
To train a SelfDriveModel use the **/SupervisedLearning/SL_self_drive_pipeline.py**. </br>
To train a DirectionClassificationModel use **/SupervisedLearning/Classification/my_classifier.py**. </br></br>

To visalize a combination model using a camera or .avi use **/SupervisedLearning/ComboNetwork/deploy_combo.py**. </br>
To visualize a combination model using a csv file use **/SupervisedLearning/ComboNetwork/visualize_combo.py**. </br>
To visualize a DirectionClassificationModel using a folder use **/SupervisedLearning/Classification/visualize_classification.py**. </br>
To visualize a SelfDriveModel using a csv file use **/SupervisedLearning/visualize_actions.py**.

The files model.cpp and model.h are NOT used in the machine learning approach. They were used in an attempt to use python models in C++.