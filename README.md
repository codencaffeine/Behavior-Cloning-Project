# Behavior-Cloning-Project

### Prerequisites
1. Linux OS (preferably Ubuntu)
2. Download simulator [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Term1-Sim/term1-simulator-linux.zip)
3. Miniconda [here](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh)
4. run `conda env create -f environment-gpu.yml` to create environment and install dependencies
5. Download Data from [data file link](https://duckduckgo.com) or pickle from [pickle file link]()


# End to End Behavioral clonning 
## Abstract
### This project presents a imitation learning based approach for autonomous driving.
Autonomous driving has been a topic of extensive research in the last few decades and many approaches has been deployed to solve the problems arising from having an unmanned vehicle. One such approach gained popularity a few years ago when Nvidia proposed a model that was not based on the standard robotics based approach but on ent to end deep learning. This approach gave promising results when they drove an actual car on the roads they trained it on. 
The Main challenges today for autonomous cars is conditions like unmarked or unpaved roads, or the time of the day when the reflection of the sun rays affect the visibility of the road lines, etc. In such conditions, the systems based on lane detection face a high chance of failure and have not been able to perform robustly. The approach thus has an advantage in such cases as it does not deploy lane detection and can drive successfully even through unpaved roads, parking lots etc.


<!-- ![alt text](./hist_before.png "Title") -->
## Motivation

The process is based on the end to end behavioral cloning for self driving cars by Nvidia [1].  The simulator used for the purpose of implementing this project is provided by Udacity. There are two modes after installation: training mode and autonomous mode.
Training mode is to collect the data and autonomous mode is to see whether our car runs automatically after the model is trained

## Implementation
![alt text](./pop.png "Title")
1. As shown in the diagram above, the car is first driven on a track using the simulator and the steering angles along with the images captured by the three front cameras of the car are recorded and saved as a csv file. We can create as many data samples as we want for different scenarios by driving on the same chosen track multiple times.
The model will be only as good as the data is, so it is important to drive along the track without any agreesive turns and must be mostly at the center of the road.

2. The first step is to read the data from the csv file. After which the data has to be preprocessed befor it can be given to the model.

3. For preprocessing, The images are flipped and the corresponding steering angles are negated. For the left and right camera feeds, the steering angles assigned to them will be offsetted by (+ or - 0.2) value so that it doesnt go off the track.
4. To visualise the data, histogram was plotted and it was found to have a few dominating bins. 

![alt text](./hist_before.png "Title")

5. This was corrected by resampling the data to its men by randomly delecting the data above the mean value and the resulting histogram had a more uniform look.
6. The images fed into the model are then cropped from the top and bottom to remove the information that are needless for the model. It is shown by the red demarcation line.
   
7.  After which, the images values were normalised.
8.  I am using a sequential CNN model programmed using the keras API along with the loss function as MSE. The optimizer is Adam and the activation function is Rectifier Linear Unit.
9.  The learning rate was set at (10<sup>-4</sup>) and the model was trained in just 5 epochs. The model might not perform smoothly above 5 epochs probably due to the over fitting issue.
10.  As for future scope, this project has a better scope with reinforcement learning.

## Evaluation

1. The code was written based on experimentation and what led to the smoothest ride on the track. this was done by letting the car drive autonomously after each time the model was trained
2. The car was made to run multiple laps both in the forward and the reverse directions. 
3. This model when run with the simulator made the car run perfectly along the center of the road without ever touching the lanelines.

## References
[1] Bojarski, Mariusz, D. Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, L. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, X. Zhang, Jake Zhao and Karol Zieba. “End to End Learning for Self-Driving Cars.” ArXiv abs/1604.07316 (2016): n. pag.
