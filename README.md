# Active Data Selection
This repository contains a series of demonstrations of some simple examples of active data selection. The core idea is that we can appeal to the foveal sampling strategies in biological visual search to motivate the selection of small subsets of data, either from a very large dataset, or through carefully motivated experiments. The key is selection of choices $\pi$ that optimise the mutual information $I$ between data $y$ and their causes $\theta$:

$$ I[\pi] = \mathbb{E}\left[ P(y,\theta|\pi)||P(y|\pi)P(\theta|\pi) \right] $$

This depends upon the form of generative model $P(y,\theta|\pi)$ under which data are assumed to be generated. The demo routines here illustrate this for models that allow for simple function approximation, dynamically evolving functions, and in more complex models of the sort we might expect in clinical trial design.
