A PyTorch implementation of a conditional diffusion model (DDPM) trained on a synthetic 2D dataset.

Built during my internship to learn how diffusion models work, with guidance from my mentor and assigned tasks.

The dataset consists of 5 Gaussian clusters in 2D, each labeled with flag 0 or flag 1. 
The model learns to reverse a gradual noising process and, once trained, generates new 2D samples conditioned on the desired flag.

Pipeline: forward diffusion (adding noise) → denoising network (predicting noise, conditioned on timestep + flag) → reverse diffusion (sampling new data from pure noise).

How to run: open 2D_Feedforward_Model.ipynb in Colab or Jupyter (requires torch, numpy, matplotlib, tqdm) and run all cells.
