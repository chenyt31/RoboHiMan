# RoboHiMan

RoboHiMan is a hierarchical evaluation paradigm for compositional generalization in long-horizon manipulation. RoboHiMan introduces HiMan-Bench, a benchmark of atomic and compositional tasks under diverse perturbations, supported by a multi-level training dataset for analyzing progressive data scaling, and proposes three evaluation paradigms (vanilla, decoupled, coupled) that probe the necessity of skill composition and reveal bottlenecks in hierarchical architectures. Experiments highlight clear capability gaps across representative models and architectures, pointing to directions for advancing models better suited to real-world long-horizon manipulation tasks. Website: [https://robohiman.github.io/](https://robohiman.github.io/)

- Installation: [INSTALL.md](INSTALL.md)
- Baselines:
    - [3d-Diffuser-Actor-Baseline](baselines/3d-Diffuser-Actor-Baseline/README.md)
    - [OpenPi-Baseline](baselines/OpenPi-Baseline/README.md)
    - [OpenPi05-Baseline](baselines/OpenPi05-Baseline/README.md)
    - [RVT-Baseline](baselines/RVT-Baseline/README.md)

# Citation
If you find this work useful, please consider citing:
```
@inproceedings{anonymous2025robohiman,
  title={RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation},
  author={Anonymous},
  booktitle={Under Review},
  year={2025}
}
```

# Acknowledgment
This project is based on the following repositories:
- [RVT](https://github.com/nvlabs/rvt)
- [3D-Diffuser-Actor](https://github.com/nickgkan/3d_diffuser_actor)
- [OpenPi](https://github.com/Physical-Intelligence/openpi)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Colosseum](https://github.com/robot-colosseum/robot-colosseum)
- [DeCo](https://deco226.github.io/)