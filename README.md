# Equivariant Manifold Flows

We provide the code for [Equivariant Manifold Flows](https://arxiv.org/abs/2107.08596) in this repository.

Summary: We lay the theoretical foundations for learning symmetry-invariant distributions on arbitrary manifolds via our introduced equivariant manifold flows. We demonstrate the utility of our approach by using it to learn gauge invariant densities over SU(n), useful for lattice quantum field theory.

Example learned densities, together with baselines, are given below.

SU(2)             |  SU(3)
:-------------------------:|:-------------------------:
![SU(2)](https://i.imgur.com/FrDmsBG.png)| ![SU(3)](https://i.imgur.com/h2LJWDd.png)

Sphere            |
:-------------------------:
![Sphere S^2](https://i.imgur.com/gd3IeGF.png)|

Below we have visualized how our equivariant manifold flows learn the `boyda1` and `boyda2` densities on SU(3) (first and second rows in the SU(3) figure above).

SU(3) Boyda1             |  SU(3) Boyda2 
:-------------------------:|:-------------------------:
![SU(3) Boyda1](https://media3.giphy.com/media/Q0tfEXosPSn7SlSgKR/giphy.gif)| ![SU(3) Boyda2](https://media.giphy.com/media/7p3PBzoD5XYdfwCFfK/giphy.gif)

## Software Requirements
This codebase requires Python 3 and PyTorch 1.8+.

## Usage

### Demo

The following command learns a equivariant manifold flow model for the `boyda2` density on SU(3). The density learned will be symmetric with respect to conjugation by SU(3), which we prove is an isometry in Appendix A.5 of our paper.

```
python main.py --M SU3 --batch_size 1024 --dev cpu --save_viz --contsave --save_freq 1 --lr 4e-3 --epochs 200 --example_density boyda2
```

Note that 1024 samples are used per epoch. 200 epochs were used for the results in the paper. The learned density is given below, in comparison with the ground truth density:

Ground Truth             |  Ours (50 epochs)         |    Ours (200 epochs)
:-------------------------:|:-------------------------:|:---------------------:
![SU(3) GT](https://i.imgur.com/0SNBMQe.png)| ![SU(3) 50 epochs](https://i.imgur.com/RDSe1O7.png) | ![SU(3) 200 epochs](https://i.imgur.com/oVNKrxo.png)

Observe that even after 50 epochs (only approximately 50,000 samples), our model approaches the ground truth.

### Full Usage

All options are given below:

```
usage: main.py [-h] [--example_density EXAMPLE_DENSITY] [--num_drops NUM_DROPS] [--save_viz] [--save_model] [--dev DEV]
               [--M {SU3}] [--contsave] [--save_freq SAVE_FREQ] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
               [--weight_decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  --example_density EXAMPLE_DENSITY
                        example density to learn. Options are: boyda1 | boyda2 | boyda3 (different densities defined on
                        SU(3))
  --num_drops NUM_DROPS
                        number of times to drop the learning rate
  --save_viz            Save a visualization of the learned density once training is completed
  --save_model          Save a visualization of the learned sampler
  --dev DEV             Default device; cpu preferred due to poor support of complex tensor differentiation on cuda
  --M {SU3}             Manifold over which to learn
  --contsave            Continuously save intermediate flow visualization in contsave/
  --save_freq SAVE_FREQ
                        frequency of continuous saving of intermediate flows
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
```

## Attribution

If you use this code or our results in your research, please cite:

```
@article{Katsman2021EquivariantMF,
  title={Equivariant Manifold Flows},
  author={Isay Katsman and Aaron Lou and D. Lim and Qingxuan Jiang and Ser-Nam Lim and Christopher De Sa},
  journal={ArXiv},
  year={2021},
  volume={abs/2107.08596}
}
```
