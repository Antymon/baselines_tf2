Baselines TF2.0
==========
Selected Deep RL algorithms using Tensorflow 2.0's Autograph and `tf.function` which are described as a merger of 
a priori computational graph definition and eager execution. Algorithms were largely ported from Stable Baselines @Hill2018 Deep Reinforcement Learning suite (with some utils brought over).
At the moment SAC @Haarnoja2018 and DDPG @Silver2014 are available.

Prerequisites
--------
Tensorflow 2.0 is needed. As of November 2019 this is a default version when pulled through **latest** `pip`.

Examples
--------
There are 2 tested SAC hyperparameter configurations in a form of `run_sac_*.py` under the root directory. 
Those are: a) simple Continuous Lunar Lander environment from OpenAI Gym framework @Brockman2016, 
b) more complicated 18-DOF Hexapod robot setup through DART simulation engine @Lee2018. Hexapod robot @Cully2015 is tasked to walk as far as possible along X-axis ([example recording](https://drive.google.com/open?id=1ds_VrjTDdhqWkh40eF1vscetfUyJUlVm)).


1) To set up Lunar Lander the OpenAI's Gym with Box2D is needed (can be found at respective pages, helper script for the latter under Linux is provided). Solved environment yields reward of 200 and above.

2) Setup of a Hexapod is more involved - I can publish a Docker Linux container at a request (based on repositories mentioned below). Generally a reward of under 2 signifies hexapod properly walking. TensorBoard charts obtained from the example show that SAC can reach that state in just 2M frames (and hyper parameters were not tuned at all):

![SAC_hexapod_training](SAC_hexapod_cl.png)


State of the project
--------
This a quick proof-of-concept set up for educational purposes. SAC should be easy to follow although generally codebase is not High Valyrian. Let me know if this project is useful for you!

<a name="repos"></a>Related repositories
==========
-   [docker-pydart2\_hexapod\_baselines](https://gitlab.doc.ic.ac.uk/sb5817/docker-dart-gym) - Docker @Merkel2014 file describing hexapod Python setup. Would require `pip` and `tensorflow` updates to work with this repository.

-   [gym-dart\_env](https://gitlab.doc.ic.ac.uk/sb5817/dart_env) -
    Hexapod setup as a Python-based environment within OpenAI Gym
    @Brockman2016 framework.
    
-   [pydart2](https://gitlab.doc.ic.ac.uk/sb5817/pydart2) - Fork of
    Pydart2 @Ha2016: Python layer over C++-based DART @Lee2018
    simulation framework. Modified to enable experiments with hexapod.
    
References
==========
1. Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, JeffreyDean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In12th{USENIX}Sym-posium on Operating Systems Design and Implementation ({OSDI}16), pages265–283, 2016
0. Greg  Brockman,  Vicki  Cheung,  Ludwig  Pettersson,  Jonas  Schneider,  John Schulman,  Jie  Tang,  and  Wojciech Zaremba.   Openai  gym.arXiv  preprintarXiv:1606.01540, 2016
0. Antoine Cully, Jeff Clune, Danesh Tarapore, and Jean-Baptiste Mouret. Robots that can adapt like animals. Nature, 521(7553):503, 2015
0. Sehoon  Ha. Pydart2:   A python binding of  DART. https://github.com/sehoonha/pydart2, 2016
0. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).
0. Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar et al. "Soft actor-critic algorithms and applications." arXiv preprint arXiv:1812.05905 (2018).
0. Ashley Hill, Antonin Raffin, Maximilian Ernestus, Adam Gleave, Rene Traore, Prafulla Dhariwal, Christopher Hesse, Oleg Klimov, Alex Nichol, Matthias Plap-pert,  Alec Radford,  John Schulman,  Szymon Sidor,  and Yuhuai Wu.   Stablebaselines.https://github.com/hill-a/stable-baselines, 2018
0. Jeongseok Lee, Michael Grey, Sehoon Ha, Tobias Kunz, Sumit Jain, Yuting Ye, Siddhartha Srinivasa, Mike Stilman, and C Karen Liu.  Dart:  Dynamic animation and robotics toolkit.The Journal of Open Source Software, 3:500, 02 2018
0. Dirk Merkel. Docker: Lightweight Linux containers for consistent development and deployment. Linux J., 2014(239), March 2014
0. David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, and Martin Riedmiller. "Deterministic policy gradient algorithms." 2014.
