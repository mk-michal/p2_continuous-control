## Project 2: Continuous Control
# Algorithm
To solve Reacher task, we used DDPG algorithm with actor-critic mechanism. 
We reused DDPG algorithm provided by [ShangtonZhang repository](https://github.com/ShangtongZhang/DeepRL) 
with some adjustments that fitted our needs. 

DDPG was trained on a single Reacher Agent and trained on GPU in order to speed up the training.
We use convolutional network with two hidden layers of 400 and 300 units for actor and critic,
learning rate of actor was set to 1e-4 and critic to 1e-3, we used ReplayBuffer to pull
sample examples to train network and soft-updated that updated target network by some changes
in local network. We updated the network after every step we took. 

# Navigation
The DDPG model could be found in ddpg_model.py module. It receives a parent BaseAgent class
from ShangtonZhang repository. 

Training is done in `ddpg_runner.py`. Agent receives hyperparameters that are defined in `utils.py`

Final results are saved into `results` folder. We can find the trained agent model, as well
with used normalizer and graph of final results.

In order to only run trained model in evaluation model, run `model_evaluation.py`.

# Hyperparameters
Final hyperparameters:

Actor/Critic conv hidden layers: 400, 300
actor lr: 1e-4
critic lr: 1-3
target_network_mix = 0.005
discount_rate: 0.99
batch_size: 100
Memory_size: 1000000

# Trials
We tried larger networks with more convolutional layers in our first attempts, however
we stopped the training as it was taking a long time a it was converging very slowly.

We also use smaller soft update than 0.005, but our algorithm failed to reach desired score.
It stopped at around 8 points per episode a wasnt increasing from there.

Also lowering the learning rate of the actor from 1e-3 to 1e-4 helped with reaching higher scores.
This may be the final reason why we achieved final desirable result

# Results
[image]:results/results.png
![Final Results][image]


We achieved desired result after approximately 500 episodes, as we can see in the chart. 
From there, the average reward was still increasing, but we stopped the training, as it
was already running for couple of hours on GPU. 

# Possible adjustments
We could definatelly achieve faster convergence with different algorithm (PPO) or different
choice of hyperparameters. One thing we could try is to do larger updates to network. Maybe instead
of updating in each step, we could make one larger update after x steps. 

