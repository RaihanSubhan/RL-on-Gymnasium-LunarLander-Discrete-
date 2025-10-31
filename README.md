RL on Gymnasium (LunarLander Discrete)

## Project Description
This project implements a Deep Q-Network (DQN) reinforcement learning agent to solve the LunarLander-v3 environment from Gymnasium. The agent learns to successfully land a lunar lander between two flags using discrete actions.

## Algorithm
Deep Q-Network (DQN) with Experience Replay and Target Network

## Requirements

### System Requirements
- Python 3.8 or higher
- Google Colab (recommended) or local Python environment
- GPU (optional, but recommended for faster training)

### Python Package Requirements
Install the following packages before running the code:

gymnasium[box2d]
torch
matplotlib
numpy
imageio
imageio-ffmpeg

### Installation Commands

For Google Colab:
!apt-get install -y swig
!pip install gymnasium[box2d] torch matplotlib numpy imageio imageio-ffmpeg

For Local Installation (Windows):
pip install swig
pip install gymnasium[box2d] torch matplotlib numpy imageio imageio-ffmpeg

For Local Installation (Mac/Linux):
apt-get install -y swig
pip install gymnasium[box2d] torch matplotlib numpy imageio imageio-ffmpeg

## Hyperparameters
- Learning Rate: 0.001
- Gamma (Discount Factor): 0.99
- Epsilon Start: 1.0
- Epsilon End: 0.01
- Epsilon Decay: 0.995
- Batch Size: 64
- Replay Buffer Size: 10000
- Target Network Update Frequency: Every 10 episodes
- Total Training Episodes: 1000

## Network Architecture
- Input Layer: 8 neurons (state dimensions)
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 128 neurons with ReLU activation
- Output Layer: 4 neurons (action dimensions)

## Environment Details
- Name: LunarLander-v3 (Discrete)
- State Space: 8-dimensional continuous vector
  - x position, y position
  - x velocity, y velocity
  - angle, angular velocity
  - left leg contact, right leg contact
- Action Space: 4 discrete actions
  - 0: Do nothing
  - 1: Fire left orientation engine
  - 2: Fire main engine
  - 3: Fire right orientation engine
- Reward Structure:
  - Positive reward for landing successfully
  - Negative reward for crashing or moving away from landing pad
  - Target reward: 200+ for stable performance

## Files Included
- main.py - Complete training and testing code with command-line interface
- model.pth - Trained DQN model weights
- train_plot.png - Learning curve showing episode rewards vs episodes
- README.md - This documentation file

## How to Execute

### Step 1: Install Requirements
Run the installation commands listed above in the "Requirements" section.

### Step 2: Training a New Model
To train the agent from scratch:

python main.py

Optional: Specify number of training episodes
python main.py --episodes 1500

Training will:
- Run for the specified number of episodes
- Save the trained model as 'model.pth'
- Generate and save 'train_plot.png' with the learning curve
- Print episode rewards and epsilon values during training

### Step 3: Testing the Trained Model
To test and visualize the trained agent:

python main.py --test

Optional: Specify custom model path
python main.py --test --model_path path/to/model.pth

Testing will:
- Load the trained model weights
- Run 5 test episodes with rendering
- Display the agent's performance
- Print reward for each test episode

### Google Colab Specific Instructions

1. Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

2. Create project directory:
!mkdir -p "/content/drive/MyDrive/Colab Notebooks/RL_HW3"

3. Run training cells in order (Cell 1 through Cell 7)

4. Test the model using Cell 9

5. Videos will be saved to Google Drive at:
   /content/drive/MyDrive/Colab Notebooks/RL_HW3/

## Training Summary
The agent was trained for 1000 episodes using the DQN algorithm with experience replay and a separate target network. The epsilon-greedy exploration strategy starts at 1.0 (full exploration) and gradually decays to 0.01 (mostly exploitation). The learning curve demonstrates clear upward trend and convergence, with the agent achieving stable performance with average rewards above 200. This indicates successful learning of the landing task.

## Expected Results
- Training time: Approximately 30-60 minutes for 1000 episodes
- Final average reward: 200+ (successful landing)
- Learning curve: Upward trend with convergence
- Test performance: Consistent successful landings between flags

## Video Demonstration
YouTube Link: https://youtu.be/o3uPCdC5aoo

Description: 1-minute demonstration video showing the trained agent successfully landing the lunar lander with smooth, controlled descent.

## Troubleshooting

### Box2D Installation Error
If you encounter "Box2D is not installed" error:
!apt-get install -y swig
!pip install gymnasium[box2d]

### Model Not Found Error
Ensure the model.pth file exists in the correct directory. Check the path:
!ls -lh model.pth

### CUDA Out of Memory
If training fails due to memory issues, reduce batch size:
agent, rewards = train_agent(episodes=1000, batch_size=32)

## Author
[Your Name]
Student ID: [Your Student ID]
Course: Reinforcement Learning
Assignment: HW3-1

## License
This project is for educational purposes as part of coursework assignment.

## Acknowledgments
- Gymnasium (formerly OpenAI Gym) for the LunarLander environment
- PyTorch for deep learning framework
- Course instructor and TAs for guidance
