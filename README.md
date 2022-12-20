# Learning Generalizable Task Structures for TAMP

The code is organized as follows:

`main.py` file contains the key algorithm: Which includes the decoder, the learnable task embeddings, and the training procedure. Inside the problems directory the sample problems can be found. All the experiments in the figure were generated using the `problems/toy_problem.py` scenario.

To run the algorithm invoke:
`python main.py --problem toy_problem`

Refer to `python main.py -h` for the parameters and explanations.

## Additional scripts

The file `exp_generalization.py` contains the code for generalization experiments. Finally `ops` directory contains all the improvement operators,
of which only `SGLD` is currently implemented, `SGLD` is also the default improvement operator used by `main.py`.