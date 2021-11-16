## Mutli-Agent RL Stack

There are four main types of objects in our stack:
- Learners - represent learning algorithms, train asynchronously on data collected by actors
- Actors - represent policies (or collections of policies) that can be executed, also handle feeding experience to learners
- Trainers - Control the sampling/training loop, can simply feed data to a single learner, or can implement more complex self-play strategies involving multiple learner classes
- Environments - represent the learning task itself, provide access to simulators and external control interfaces.  Also defines the action and observation spaces

This stack is similar to and inspired by RLLib but with a number of key differences:
- Trainers and Learners are loosely coupled.  Single-agent algorithms will typically be implemented by a learner alone, with a common distributed Trainer.
- Better support for explicit serialization of policies for deployment/self-play.  Not completely reliant on a checkpointing mechanism to implement such functionality
- Multi-agent structure provided by environment, rather than as a separate configuration parameter.  Agent symmetry, team organization, and payoff structure are all predefined in the environment specification
- Support for auxiliary (centralized) observations and asynchronous interaction loops

Engineering notes:
- We will only support Torch as a backend for the moment
- Algorithms will be pure python, though individual environments may be compiled for better performance

### Learners

We can think of a learner as a central hub which collects experience data and uses it to update a collection of policies and associated evaluation functions.  Learners interact with their environment by generating Actor objects which serve as interfaces to the environment.  For multi-agent scenarios, a learner might represent a single policy used by one or more agents, or it might represent a collection of policies and value functions which share parameters.

Learners always implement an asynchronous, distributed training API, even when using an on-policy learning algorithm.  If synchronous training is required (e.g. policy gradients, PPO), it is up to the Trainer to enforce this.

In special cases, learners can represent auxiliary functions, such as centralized critics or curiosity models that are not tied to any single agent or policy.  Actors generated by these learners will only yield auxiliary data, and not control an agent.  Learners are also responsible for managing any static datasets (e.g. human trajectories).

### Actors

An actor is an object which takes in a stream of observations (which may come from multiple agents), and returns actions for those agents.  In the single-agent setting, an actor represents a single policy, while in a multi-agent setting, and actor represents a mapping from available agents to policies.

Actors also represent the interface between the environment and the learner.  Actors are responsible for packaging and forwarding data to the learner, and are responsible for making sure that their models are consistent with the learner's model.  In addition to action generation, actors also implement a sync() method, which triggers the transfer of data to the learner, and the download of new policies if needed.

Actors are assumed to run on a single thread, with a single learner having multiple actors for different nodes/threads.  We define separate actors for evaluation and exploration, with the learner providing methods for constructing both.  Actors are always associated with a Learner, with fixed strategies being implemented by a Learner that ignores experience, and yields actors that encapsulate pre-trained or hard-coded policies.

### Trainers

Trainers manage the entire training process, and implement a train() method which executes one "iteration" of training.  Trainers are responsible for managing distributed training across multiple threads/nodes.  As much as possible, Trainers are algorithm-agnostic, and there will be a default Trainer for all single-agent algorithms.  For more complex multi-agent scenarios, Trainers are used to implement self-play and league architectures.  Trainers are explicitly designed to be "nested" with different sub-trainers controlling different part of the training pipeline.  Trainers manage both data collection and policy evaluation.

### Environments

The environment represents an interface to a simulator, or to an external environment.  There are three main types of environments we want to support:
- Internal Environments - These are environments that are implemented by python objects, and have no external (non-python) dependencies.  Gym environments would be the most common example.  Internal environments can be instantiated at will on any thread or node, and support environment batching.
- External Simulators - Environments which are simulated, but where the simulators are outside the main process's control.  An example would be a video game which has to be launched by an external os command, potentially on a separate machine.  There may be limits on the number of instances of the simulator that can be run simultaneously, and the interaction loop will be driven by the simulator (the Trainer will not increment the environment itself).
- External Environments - These would be "real-world" environments, such as a physical robot, which are entirely outside the Trainer's control.  External environments are instantiated independently of the trainer, and the trainer is responsible for providing actions for all active environments.

Environment objects may represent multiple environments that can be accessed separately by the Trainer.  Multiple semantic structures over these environments are supported, including:
- Training/Test Splits - Allows different sets of environments for training and evaluation
- Curricula - Partial orderings over environments in terms of complexity
- Adversarial Environments - Support learners which try to maximize worst-case performance over a set of different environments
- Prior distributions - Allow environment to be sampled from some prior distribution at the start of each episode.
Environments are indexed by string IDs, with semantic structure provided separately from the individual environment specifications.  If multiple environments are specified, we require that a semantic structure be provided. 

We use a more general interaction loop and environment specification than Gym.  All environments are interpreted as being multi-agent, but each environment generates a single, hierarchical stream of observations.  The observation and action spaces are defined by a single tree data structure.  There are several types of nodes in this tree:
- Observation Nodes - Represent observation spaces.  Can take on any Gym observation space, including nested Dict spaces (enabling hierarchical structure).  Allows for components of the observation space to be marked as "evaluation only", indicating that the observations should only be used for training, not in the deployed policies.
- Action Nodes - Same as Observation nodes, but representing action spaces.  Do not allow for evaluation tags.  Does allow for "non-atomic" tags, which indicate that individual elements of a dict action space can be executed separately.
- Agent Nodes - Represent individual agents (or templates for repeated agents) children are an observation and action space
- Population Nodes - Represents a set of agents.  Children are either individual agent nodes or other population nodes.  Can additionally include common observation spaces, and metadata indicating whether the population has cooperative, zero-sum, or general payoffs.  Can take one of two forms:
    - Dictionary - a fixed set of uniquely named agents.  Represents cases where there are a fixed number of named agents.
    - Array - not really an array, but a node indicating that there are a variable number of agents with identical action and observation spaces

  
### Distributed Computing

In addition to GPU-accelerated vector operations, there will be three levels of CPU parallelism:
- Experience Sampling - simulation for evaluation and experience collection can be distributed
- Multi-learner Training - When multiple learners are present, they can be trained in parallel
- Experimental Runs - for hyperparameter tuning or testing multiple seeds, trainers can be run in parallel

The simplest way to implement this is to wrap Trainers, Learners, and simulation processes in Ray actors.  For now, we can leave it up to the Trainer to initiate remote simulations/learning processing.  While being tied to Ray is not ideal, we can likely hide the Ray API so that we can replace it later if needed.

#### Distrubuted Sampling

The main value of distributed learning is the ability to sample training data from multiple simulation workers distributed across cores/nodes.

In RLLib, this process is centralized, with a single training worker requesting a certain amount of experience data, and once this data is generated, feeding it to the learner, which typically runs on in the same process as the trainer.  This approach works well for single agent and cooperative multi-agent setups, but is not ideal for training populations of heterogeneous learning agents.

An alternative approach is to treat the sampling processes as real environments, over which individual learners have no control.  An example might be a collection of real robots that are interacting with physical environments which need to be manually reset after each episode.  The learner needs to be able to receive and interpret experience data as it arrives.  In this setting, a learner may have many associated "Actor" objects which serve as the local interfaces between the learner and individual simulations.  Actors associated with a single learner need not be identical (e.g. training vs. eval) actors, but the learner should not keep track of individual actors.  Actors should send formatted streams of data to the learner, while the learner should indicate to all actors when shared states (e.g. policy weights) should be updated.

All actors an learners should use a shared communication channel that avoids unnecessary duplication of data or unnecessary network traffic.  In a multi-agent atari environment, for example, each learner receives a separate logical stream of data, but the frames actually being accessed by different learners on the same node are backed by the same memory, while frames are only transferred once between nodes even if they are "sent" by multiple actors.

From the trainer's perspective, it instantiates the learner(s), then requests training and eval actors for each simulation worker.  The simulation workers are then run until enough data (episodes or steps) has been generated for the current training iteration.  To ensure eval actors follow a fixed policy, and to prevent unnecessary policy updates, we synchronize Actors externally by calling a .sync() method.  Therefore, Actors can send data to the learner whenever they wish, but can only receive data at intervals that are beyond their control.  Furthermore, they have no way of knowing whether their data has been uploaded to the learner itself.

In general, the learners themselves will be data-driven, performing training when sufficient amounts of data have arrived.  We will add a .run() method to allow learners to define a continuous training loop (useful for batch RL).