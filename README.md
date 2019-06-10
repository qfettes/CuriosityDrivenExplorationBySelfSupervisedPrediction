# Reproduction of "Curiosity-driven Exploration for Deep Reinforcement Learning" with PyTorch
This is a PyTorch reproduction of the ICML 2017 paper "Curiosity-driven Exploration for Deep Reinforcement Learning." It uses A2C rather than A3C. It only allows evaluation on the SuperMarioBros environment right now.
Look at the original Tensorflow implementation [here](https://github.com/pathak22/noreward-rl).

# To Train
python3 a2c_devel.py --recurrent-policy

Requirements: 
* Python 3.6
* Numpy 
* Gym 
* Pytorch >=0.4.0 
* Matplotlib 
* OpenCV 
* Baslines
* [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

Acknowledgements: 
* Credit to [@pathak22](https://github.com/pathak22/noreward-rl) For their publicly available original implementation
* Credit to [@baselines](https://github.com/openai/baselines) for templates for the environment wrappers
* Credit to [@sadeqa](https://github.com/sadeqa/Super-Mario-Bros-RL/blob/master/A3C/common/atari_wrapper.py) for the template for ProcessFrameMario() wrapper
* Credit to [@ikostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) for A2C, GAE, PPO and inspiration for plotting code implementation
