# Text-based game using reinforcement learning  

This is a project of my **MITx Machine Learning with Python** course. The objective is to address the task of learning control policies for text-based games using reinforcement learning.  

## Setup of the project  

In text-based games, all interactions between players and the virtual world are through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions.  

For this project I conducted experiments on a small **Home World**, which mimic the environment of a typical house. The world consists of a few rooms, and each room contains a representative object that the player can interact with. For instance, the kitchen has an **apple** that the player can **eat**. The goal of the player is to finish some quest. An example of a quest given to the player in text is **You are hungry now** . To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple. In this game, the room is hidden from the player, who only receives a description of the underlying room. At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., **eat apple**). The player then receives some reward that depends on the state and his/her command.  

In order to design an autonomous game player, I employed a reinforcement learning framework to learn command policies using game rewards as feedback. Since the state observable to the player is described in text, some mechanisms that maps text descriptions into vector representations are essential. A naive approach is to create a map that assigns a unique index for each text description. However, such approach becomes difficult to implement when the number of textual state descriptions are huge. An alternative method is to use a bag-of-words representation derived from the text description.  

## What I did  

1. Implemented the tabular Q-learning algorithm for a simple setting where each text description is associated with a unique index.  

2. Implemented the Q-learning algorithm with linear approximation architecture, using bag-of-words representation for textual state description.  

3. Implemented a deep Q-network using pytorch. A neural network was trained to map text described states and actions to $Q(s,c)$ function.

4. Experimented Q-learning algorithms on the Home World game with different hyper-parameters. 

## Key notes

1. The agent plays an action $c$ at state $s$, getting a reward $R(s, c)$ and observeing the next state $s'$.  

2. Update the single Q-value corresponding to each such transition: 
$Q(s,c)\leftarrow (1-\alpha )Q(s,c)+\alpha [R(s,c)+\gamma \max _{c'\in C}Q(s',c')]$

3. Total discounted reward: $\sum _{t=0}^{\infty }\gamma ^{t}r_{t}.$  

