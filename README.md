# Scripts for training and validating neural network models for gnina

 * train.py - Takes a model template and train/test file(s)
 * predict.py - Takes a model, learned weights, and an input file and outputs probabilities of binding

## Training

```
usage: train.py [-h] -m MODEL -p PREFIX [-n NUMBER] [-i ITERATIONS] [-s SEED]
                [-t TEST_INTERVAL] [-o OUTPREFIX] [-g GPU] [-c CONT] [-k] [-r]
                [--avg_rotations] [--keep_best] [--dynamic] [--solver SOLVER]
                [--lr_policy LR_POLICY] [--step_reduce STEP_REDUCE]
                [--step_end STEP_END] [--step_when STEP_WHEN]
                [--base_lr BASE_LR] [--momentum MOMENTUM]
                [--weight_decay WEIGHT_DECAY] [--gamma GAMMA] [--power POWER]
                [--weights WEIGHTS]

Train neural net on .types data.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model template. Must use TRAINFILE and TESTFILE
  -p PREFIX, --prefix PREFIX
                        Prefix for training/test files:
                        <prefix>[train|test][num].types
  -n NUMBER, --number NUMBER
                        Fold number to run, default is all
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations to run,default 10,000
  -s SEED, --seed SEED  Random seed, default 42
  -t TEST_INTERVAL, --test_interval TEST_INTERVAL
                        How frequently to test (iterations), default 40
  -o OUTPREFIX, --outprefix OUTPREFIX
                        Prefix for output files, default <model>.<pid>
  -g GPU, --gpu GPU     Specify GPU to run on
  -c CONT, --cont CONT  Continue a previous simulation from the provided
                        iteration (snapshot must exist)
  -k, --keep            Don't delete prototxt files
  -r, --reduced         Use a reduced file for model evaluation if exists(<pre
                        fix>[_reducedtrain|_reducedtest][num].types)
  --avg_rotations       Use the average of the testfile's 24 rotations in its
                        evaluation results
  --keep_best           Store snapshots everytime test AUC improves
  --dynamic             Attempt to adjust the base_lr in response to training
                        progress
  --solver SOLVER       Solver type. Default is SGD
  --lr_policy LR_POLICY
                        Learning policy to use. Default is inv.
  --step_reduce STEP_REDUCE
                        Reduce the learning rate by this factor with dynamic
                        stepping, default 0.5
  --step_end STEP_END   Terminate training if learning rate gets below this
                        amount
  --step_when STEP_WHEN
                        Perform a dynamic step (reduce base_lr) when training
                        has not improved after this many test iterations,
                        default 10
  --base_lr BASE_LR     Initial learning rate, default 0.01
  --momentum MOMENTUM   Momentum parameters, default 0.9
  --weight_decay WEIGHT_DECAY
                        Weight decay, default 0.001
  --gamma GAMMA         Gamma, default 0.001
  --power POWER         Power, default 1
  --weights WEIGHTS     Set of weights to initialize the model with
```

MODEL is a caffe model file and is required.

PREFIX is the prefix of pre-specified training and test files.  For example, 
if the prefix is "all" then there should be files "alltrainX.types" and
"alltestX.types" for X from 0 to N, where N is the number of folds.
The model trained on "alltrain0.types" will be tested on "alltest0.types"
every TEST_INTERVAL iterations.

The train/test files are of the form
    1 set1/159/rec.gninatypes set1/159/docked_0.gninatypes
where the first value is the label, the second the receptor, and the third
the ligand.  Additional whitespace delimited fields are ignored.


The prefix of all generated output files will be OUTPREFIX.  If not specified,
it will be the model name and the process id.  While running, OUTPREFIX.X.out files are generated.
A line is written at every TEST_INTERVAL and consists of the test auc, the train auc, the loss,
and the learning rate at that iteration.

The entire train and test sets are evaluated every TEST_INTERVAL.  If they are very large
this may be undesirable.  Alternatively, if -r is passed, pre-specified reduced train/test sets
can be used for monitoring.

Once each fold is complete OUTPREFIX.X.finaltest is written with the final predicted values.

After the completion of all folds, OUTPREFIX.test and OUTPREFIX.train are written which
contain the average AUC and and individual AUCs for each fold at eash test iteration.
Also, a total finaltest file of all the predictions.  Graphs of the training behavior
(OUTPREFIX_train.pdf) and final ROC (OUTPREFIX_roc.pdf) are also created as well as caffe files.

The GPU to use can be specified with -g.

Previous training runs can be continued with -c.  The same prefix etc. should be used.

A large number of training hyperparameters are available as options.  The defaults should be pretty reasonable.

