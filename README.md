# Scripts for training and validating neural network models for gnina

 * train.py - Takes a model template and train/test file(s)
 * predict.py - Takes a model, learned weights, and an input file and outputs probabilities of binding
 * clustering.py - Takes an input file, and outputs clustered cross-validation train/test file(s) for train.py. Note: take a long time to compute
 * compute_seqs.py - Takes input file for clustering.py and creates the input for compute_row.py
 * compute_row.py - Computes 1 row of NxN matrix in clustering.py. This allows for more parallelization of clustering.py
 * combine_rows.py - Script to take outputs of compute_row.py & combine them to avoid needing to do the computation in clustering.py
 * simple_grid_visualization.py - Script to evaluate 2-D single ligatom+receptor atom systems to visualize how CNN score changes with distance between atoms.
 * grid_visualization.py - Script to evaluate a box of single atoms around a receptor of interest to visualize how CNN scores ligand contexts
 * generate_unique_lig_poses.py - Script for counter-example generation which computes all of the unique ligand poses in a directory
 * counterexample_generation_jobs.py - Script which generates a file containing all of the gnina commands to generate new counter-examples
 * generate_counterexample_typeslines.py - Script which generates a file containing the lines to add to the types file for a pocket.
 
## Dependencies

```
sudo pip install matplotlib scipy sklearn scikit-image protobuf psutil numpy seaborn
export PYTHONPATH=/usr/local/python:$PYTHONPATH
```
rdkit -- see installation instructions [here](https://www.rdkit.org/docs/Install.html)

openbabel -- see installation instructions [here](http://openbabel.org/wiki/Category:Installation)

## Training
```
usage: train.py [-h] -m MODEL -p PREFIX [-d DATA_ROOT] [-n FOLDNUMS] [-a]
                [-i ITERATIONS] [-s SEED] [-t TEST_INTERVAL] [-o OUTPREFIX]
                [-g GPU] [-c CONT] [-k] [-r] [--avg_rotations] [--keep_best]
                [--dynamic] [--cyclic] [--solver SOLVER] [--lr_policy LR_POLICY]
                [--step_reduce STEP_REDUCE] [--step_end STEP_END]
                [--step_when STEP_WHEN] [--base_lr BASE_LR]
                [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                [--gamma GAMMA] [--power POWER] [--weights WEIGHTS]
                [-p2 PREFIX2] [-d2 DATA_ROOT2] [--data_ratio DATA_RATIO]

Train neural net on .types data.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model template. Must use TRAINFILE and TESTFILE
  -p PREFIX, --prefix PREFIX
                        Prefix for training/test files:
                        <prefix>[train|test][num].types
  -d DATA_ROOT, --data_root DATA_ROOT
                        Root folder for relative paths in train/test files
  -n FOLDNUMS, --foldnums FOLDNUMS
                        Fold numbers to run, default is '0,1,2'
  -a, --allfolds        Train and test file with all data folds,
                        <prefix>.types
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
  --cyclic		Vary base_lr between fixed values based on test 
			iteration
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
  -p2 PREFIX2, --prefix2 PREFIX2
                        Second prefix for training/test files for combined
                        training: <prefix>[train|test][num].types
  -d2 DATA_ROOT2, --data_root2 DATA_ROOT2
                        Root folder for relative paths in second train/test
                        files for combined training
  --data_ratio DATA_RATIO
                        Ratio to combine training data from 2 sources
  --test_only           Don't train, just evaluate test nets once
```

MODEL is a caffe model file and is required. It should have a MolGridDataLayer
for each phase, TRAIN and TEST. The source parameter of these layers should
be placeholder values "TRAINFILE" and "TESTFILE" respectively.

PREFIX is the prefix of pre-specified training and test files.  For example, 
if the prefix is "all" then there should be files "alltrainX.types" and
"alltestX.types" for each X in FOLDNUMS.
FOLDNUMS is a comma-separated list of ints, for example 0,1,2.
With the --allfolds flag set, a model is also trained and tested on a single file
that hasn't been split into train/test folds, for example "all.types" in the
previous example.

The model trained on "alltrain0.types" will be tested on "alltest0.types".
Each model is trained for up to ITERATIONS iterations and tested each TEST_INTERVAL
iterations.

The train/test files are of the form
    1 set1/159/rec.gninatypes set1/159/docked_0.gninatypes
where the first value is the label, the second the receptor, and the third
the ligand.  Additional whitespace delimited fields are ignored.  gninatypes
files are created using gninatyper.
The receptor and label paths in these files can be absolute, or they can be
relative to a path provided as the DATA_ROOT argument. To use this option,
 the root_folder parameter in each MolGridDataLayer of the model file should be
the placeholder value "DATA_ROOT". This can also be hard-coded into the model.



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

## Generating Clustered-Cross Validation splits of data

There are 2 strategies: 1) Running clustering.py directly; 2) Running compute_seqs.py --> compute_row.py --> combine_rows.py.
Strategy 1 works best when there is a small number of training examples (around 4000), but the process is rather slow.
Strategy 2 is to upscale computing the clusters (typically for a supercomputing cluster) where each row would correspond to 1 job.

Note that these scripts assume that the input files point to a relative path from the current working directory.

### Case 1: Using clustering.py
```
usage: clustering.py [-h] [--pdbfiles PDBFILES] [--cpickle CPICKLE] [-i INPUT]
                     [-o OUTPUT] [-c CHECK] [-n NUMBER] [-s SIMILARITY]
                     [-s2 SIMILARITY_WITH_SIMILAR_LIGAND]
                     [-l LIGAND_SIMILARITY] [-d DATA_ROOT] [--posedir POSEDIR]
                     [--randomize RANDOMIZE] [-v] [--reduce REDUCE]

create train/test sets for cross-validation separating by sequence similarity
of protein targets and rdkit fingerprint similarity

optional arguments:
  -h, --help            show this help message and exit
  --pdbfiles PDBFILES   file with target names, paths to pbdfiles of targets,
                        paths to ligand smile (separated by space)
  --cpickle CPICKLE     cpickle file for precomputed distance matrix and
                        ligand similarity matrix
  -i INPUT, --input INPUT
                        input .types file to create folds from, it is assumed
                        receptors in pdb named directories
  -o OUTPUT, --output OUTPUT
                        output name for clustered folds
  -c CHECK, --check CHECK
                        input name for folds to check for similarity
  -n NUMBER, --number NUMBER
                        number of folds to create/check. default=3
  -s SIMILARITY, --similarity SIMILARITY
                        what percentage similarity to cluster by. default=0.5
  -s2 SIMILARITY_WITH_SIMILAR_LIGAND, --similarity_with_similar_ligand SIMILARITY_WITH_SIMILAR_LIGAND
                        what percentage similarity to cluster by when ligands
                        are similar default=0.3
  -l LIGAND_SIMILARITY, --ligand_similarity LIGAND_SIMILARITY
                        similarity threshold for ligands, default=0.9
  -d DATA_ROOT, --data_root DATA_ROOT
                        path to target dirs
  --posedir POSEDIR     subdir of target dirs where ligand poses are located
  --randomize RANDOMIZE
                        randomize inputs to get a different split, number is
                        random seed
  -v, --verbose         verbose output
  --reduce REDUCE       Fraction to sample by for reduced files. default=0.05
```
INPUT is a types file that you want to create clusters for

Either CPICKLE or PDBFILES needs to be input for the script to work.

PDBFILES is a file of target_name, path to pdbfile of target, and path to the ligand smile (separated by space)
CPICKLE is either the dump from running clustering.py one time, or the output from Case 2 (below) and allows you
to avoid recomputing the costly protein sequence and ligand similarity matrices needed for clustering.

When running with PDBFILES, the script will output PDBFILES.pickle which contains (distanceMatrix, target_names, ligansim),
where distanceMatrix is the matrix of cUTDM2 distance between the protein sequences,
target_names is the list of targets, and ligandsim is the matrix of ligand similarities.

When running with CPICKLE, only the new .types files will be output.

A typical usage case would be to create 5 different seeds of 5fold cross-validation.
First, we create seed0, which also will compute the matrices needed. This depends on having
INPUT a types file that we want to generate clusters for
PDBFILES (target_name path_to_pdb_file path_to_ligand_smile) for each target in types
```
clustering.py --pdbfiles my_info --input my_types.types --output my_types_cv_seed0_ --randomize 0 --number 5
```
Next we run the following four commands to generate the other 4 seeds
```
clustering.py --cpickle matrix.pickle --input my_types.types --output my_types_cv_seed1_ --randomize 1 --number 5
clustering.py --cpickle matrix.pickle --input my_types.types --output my_types_cv_seed2_ --randomize 2 --number 5
clustering.py --cpickle matrix.pickle --input my_types.types --output my_types_cv_seed3_ --randomize 3 --number 5
clustering.py --cpickle matrix.pickle --input my_types.types --output my_types_cv_seed4_ --randomize 4 --number 5
```

### Case 2: Running compute_*.py pipeline
First, we will use the compute_seqs.py to generate the needed input files
```
usage: compute_seqs.py [-h] --pdbfiles PDBFILES [--out OUT]

Output the needed input for compute_row. This takes the format of
"<target_name> <ligand smile> <target_sequence>" separated by spaces

optional arguments:
  -h, --help           show this help message and exit
  --pdbfiles PDBFILES  file with target names, paths to pbdfiles of targets,
                       and path to smiles file of ligand (separated by space)
  --out OUT            output file (default stdout)

```

PDBFILES is the same input that would be given to clustering.py.

For the rest of this pipeline, I will consider the output of compute_seqs.py to be comp_seq_out.

Second, we will run compute_row.py for each line in the output of compute_seqs.py
```
usage: compute_row.py [-h] --pdbseqs PDBSEQS -r ROW [--out OUT]

Compute a single row of a distance matrix and ligand similarity matrix from a
pdbinfo file.

optional arguments:
  -h, --help         show this help message and exit
  --pdbseqs PDBSEQS  file with target names, ligand smile, and sequences
                     (chains separated by space)
  -r ROW, --row ROW  row to compute
  --out OUT          output file (default stdout)

```
Here PDBSEQS is the output of compute_seqs.py. For example, to compute row zero
and store the output into the file row0:
```
compute_row.py --pdbseqs comp_seq_out --row 0 --out row0
```
For the next part, I assume that the output of compute_row.py is row[num] where [num] is
the row that was computed.

Third, we will run combine_rows.py to create the cpickle file needed for input into clustering.py
```
combine_rows.py row*
```
combine_rows.py accepts any number of input files, and outputs matrix.pickle

Lastly, we run clustering.py as follows
```
clustering.py --cpickle matrix.pickle --input my_types.types --output my_types_cv_
```
## Generating new counterexamples
There are 3 scripts here which form a pipeline to generate new counter-examples for a data directory.

The pipeline is as follows: 1) generate_unique_lig_poses.py; 2) counterexample_generation_jobs.py; 3) generate_counterexample_typeslines.py.

Global Assumptions: 1) The data directory structure is <ROOT>/<POCKET>/<FILES>, 2) Crystal ligand files are named <PDBid>_<ligname><CRYSTAL SUFFIX>,
	3) Receptors are PDB files, 4) output poses are SDF files.

### Step 1) Generating the unique poses for a Pocket
In order to avoid extra calculations, we need to find the unique poses.

WARNING -- this script performs an O(n^2) calcualtion for each unique ligand name in the pocket!!

This can cause this to run very slowly if there are many receptors for the ligand to be docked into.
It also can cause problems if there are many crystal ligands for 1 ligand name. 
In this case, we recommend creating a subdirectory for that pocket, and putting the extra crystal ligand files there.

```
usage: generate_unique_lig_poses.py [-h] -p POCKET -r ROOT [-ds DOCKED_SUFFIX]
                                    [-cs CRYSTAL_SUFFIX] -os OUT_SUFFIX
                                    [--unique_threshold UNIQUE_THRESHOLD]

Create ligname<OUTSUFFIX> files for use with generate_counterexample_typeslines.py.

optional arguments:
  -h, --help            show this help message and exit
  -p POCKET, --pocket POCKET
                        Name of the pocket that you will be generating the
                        file for.
  -r ROOT, --root ROOT  PATH to the ROOT of the pockets.
  -ds DOCKED_SUFFIX, --docked_suffix DOCKED_SUFFIX
                        Expression to glob docked poses. These contain the
                        poses that need to be uniqified. Default is
                        "_tt_docked.sdf"
  -cs CRYSTAL_SUFFIX, --crystal_suffix CRYSTAL_SUFFIX
                        Expression to glob the crystal ligands. Default is
                        "_lig.pdb"
  -os OUT_SUFFIX, --out_suffix OUT_SUFFIX
                        End of the filename for LIGNAME<OUTSUFFIX>. This will
                        be the --old_unique_suffix for
                        generate_counterexample_typeslines.py.
  --unique_threshold UNIQUE_THRESHOLD
                        RMSD threshold for unique poses. IE poses with RMSD >
                        thresh are considered unique. Defaults to 0.25.
```
The workflow for this script is to first generate a list of the pockets that you wish to analyze.
We provide the pockets used for our CrossDocked2020 models in cd2020_pockets.txt.

You can then run the script in a bash for loop:
```
for d in `cat cd2020_pockets.txt` do python3 generate_unique_lig_poses.py -p $d -r MYROOT -os _initial_unique.sdf; done
```
WARNING -- this will be VERY SLOW. We HIGHLY RECOMMEND running this in a parallel fashion on a super computing cluster if possible.

The output when completed will be a series of LIGNAME_initial_unique.sdf files in each pocket directory.

### Step 2 -- Generating the gnina commands to generate the counterexamples
We need to create the gnina commands to generate the new counterexample poses.
```
usage: counterexample_generation_jobs.py [-h] -o OUTFILE [-r ROOT]
                                         [-ri REC_ID] [-cs CRYSTAL_SUFFIX]
                                         [-ds DOCKED_SUFFIX] -i ITERATION
                                         [--num_modes NUM_MODES] [--cnn CNN]
                                         [--cnn_model CNN_MODEL]
                                         [--cnn_weights CNN_WEIGHTS]
                                         [--seed SEED] [--dirs DIRS]

Create cnn_minimize jobs for a dataset. Assumes dataset file structure is
<ROOT>/<Identifier>/<FILES>

optional arguments:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        Name for gnina job commands output file.
  -r ROOT, --root ROOT  ROOT for data directory structure. Defaults to current
                        working directory.
  -ri REC_ID, --rec_id REC_ID
                        Regular expression to identify the receptor PDB.
                        Defaults to ...._._rec.pdb
  -cs CRYSTAL_SUFFIX, --crystal_suffix CRYSTAL_SUFFIX
                        Expresssion to glob the crystal ligand PDB. Defaults
                        to _lig.pdb. Assumes filename is
                        PDBid_LignameLIGSUFFIX
  -ds DOCKED_SUFFIX, --docked_suffix DOCKED_SUFFIX
                        Expression to glob docked poses. These contain the
                        poses that need to be minimized. Default is
                        "_tt_docked.sdf"
  -i ITERATION, --iteration ITERATION
                        Sets what iteration number we are doing. Adds
                        _it#_docked.sdf to the output file for the gnina job
                        line.
  --num_modes NUM_MODES
                        Sets the --num_modes argument for the gnina command.
                        Defaults to 20.
  --cnn CNN             Sets the --cnn command for the gnina command. Defaults
                        to dense. Must be dense, general_default2018, or
                        crossdock_default2018.
  --cnn_model CNN_MODEL
                        Override --cnn with a user provided caffe model file.
                        If used, requires the user to pass in a weights file
                        as well.
  --cnn_weights CNN_WEIGHTS
                        The weights file to use with the supplied caffemodel
                        file.
  --seed SEED           Seed for the gnina commands. Defaults to 42
  --dirs DIRS           Supplied file containing a subset of the dataset (one pocket per line).
                        Default behavior is to do every directory.
```
The default behavior is to generate the output file for each directory in ROOT. For CrossDocked2020, we supply more pockets than we used to analyze, so you can pass the cd2020_pockets.txt file in the DIRS argument. The default values match what we used for CrossDocked2020.

Example -- generate the file it3_to_run.txt for CrossDocked2020 available at my_root for iteration 3, using the built-in dense net in gnina
```
python3 counter_example_generation_jobs.py -o it3_to_run.txt -r MYROOT -i 3 --cnn dense --dirs cd2020_pockets.txt
```
Once this has been created, each of these commands will need to be executed. Note, there will be many commands to run, so we recommend running in parallel across many GPUs on a computing cluster if possible.
### Step 3 -- Generating the lines to add to the types files.
Running this script will generate OUTNAME in the supplied pocket, which will contain the lines to add to the types files for that pocket.

WARNING -- This script also performs an O(n^2) calculation per unique ligand in the pocket! This can take a very long time, and scales with the number of receptors and crystal ligands with the same ligand name present in the pocket. If this is to much, we recommend a downsampling strategy by moving files into a sub-directory prior to running the pipeline.

WARNING 2 -- As the calculations are O(n^2), we recommend running each pocket as its own job on a computing cluster if possible.

```
usage: generate_counterexample_typeslines.py [-h] -p POCKET -r ROOT -i INPUT
                                             [-cs CRYSTAL_SUFFIX]
                                             [--old_unique_suffix OLD_UNIQUE_SUFFIX]
                                             [-us UNIQUE_SUFFIX]
                                             [--unique_threshold UNIQUE_THRESHOLD]
                                             [--lower_confusing_threshold LOWER_CONFUSING_THRESHOLD]
                                             [--upper_confusing_threshold UPPER_CONFUSING_THRESHOLD]
                                             -o OUTNAME [-a AFFINITY_LOOKUP]

Create lines to add to types files from counterexample generation. Assumes
data file structure is ROOT/POCKET/FILES.

optional arguments:
  -h, --help            show this help message and exit
  -p POCKET, --pocket POCKET
                        Name of the pocket that you will be generating the
                        lines for.
  -r ROOT, --root ROOT  PATH to the ROOT of the pockets.
  -i INPUT, --input INPUT
                        File that is output from
                        counterexample_generation_jobs.py
  -cs CRYSTAL_SUFFIX, --crystal_suffix CRYSTAL_SUFFIX
                        Expresssion to glob the crystal ligand PDB. Defaults
                        to _lig.pdb. Needs to match what was used with
                        counterexample_generation_jobs.py
  --old_unique_suffix OLD_UNIQUE_SUFFIX
                        Suffix for the unique ligand sdf file from a previous
                        run. If set we will load that in and add to it.
                        Default behavior is to generate it from provided input
                        file.
  -us UNIQUE_SUFFIX, --unique_suffix UNIQUE_SUFFIX
                        Suffix for the unique ligand sdf file for this run.
                        Defaults to _it1___.sdf. One will be created for each
                        ligand in the pocket.
  --unique_threshold UNIQUE_THRESHOLD
                        RMSD threshold for unique poses. IE poses with RMSD >
                        thresh are considered unique. Defaults to 0.25.
  --lower_confusing_threshold LOWER_CONFUSING_THRESHOLD
                        CNNscore threshold for identifying confusing good
                        poses. Score < thresh & under 2RMSD is kept and
                        labelled 1. 0<thresh<1. Default 0.5
  --upper_confusing_threshold UPPER_CONFUSING_THRESHOLD
                        CNNscore threshold for identifying confusing poor
                        poses. If CNNscore > thresh & over 2RMSD pose is kept
                        and labelled 0. lower<thresh<1. Default 0.9
  -o OUTNAME, --outname OUTNAME
                        Name of the text file to write the new lines in. DO
                        NOT WRITE THE FULL PATH!
  -a AFFINITY_LOOKUP, --affinity_lookup AFFINITY_LOOKUP
                        File mapping the PDBid and ligname of the ligand to
                        its pK value. Assmes space delimited "PDBid ligname
                        pK". Defaults to pdbbind2017_affs.txt
```

Example -- finishing the pipeline from the previous examples for ZIPA_ECOLI_187_328_0
```
python3 generate_counterexample_typeslines.py -p ZIPA_ECOLI_187_328_0 -r MYROOT -i it3_to_run.txt -us _it3___.sdf -o it3_typeslines_toadd.txt --old_unique_suffix _initial_unique.sdf
```
The above command will be need to run for each directory in cd2020_pockets.txt.  It will create the it3_typeslines_toadd.txt in the pocket directory.

That text file contains the lines that need to be added to the training/test types files. The default values match what we used for the CrossDocked2020 paper.

## Using visualization script
There are two scripts to help you visualize how the model scores atoms: 1) simple_grid_visualization.py; 2) grid_visualization.py 

Script 1 fixes a single receptor atom, & places a lig atom along specified points on the Xaxis & score them.
This gives insight into how the model behaves in a simplified 2D coordinate system.

Script 2 creates a cube of single atom points around a specified receptor, which are all scored.
This gives insight into how the model behaves in a simplified 3D system. A glycine tripeptide is included
in this directory as a starting point.

Note: both of these scripts need to be run twice in order to complete their entire function.
### Case 1: Using simple_grid_visualization.py
```
usage: simple_grid_visualization.py [-h] -r RECATOMS -l LIGATOMS [-o OUTNAME]
                                    [-t TYPESROOT] -m MODEL -w WEIGHTS
                                    [-n NUM_POINTS] [-i INCREMENT]
                                    [-b BOX_SIZE] [--plot] [-d DATAROOT]

Script for generating the jobs that need to be run for simple visualization.
Generates types files & a text file that needs to be run. This results in
separating atoms along the x-axis. Can then graph the results.

optional arguments:
  -h, --help            show this help message and exit
  -r RECATOMS, --recatoms RECATOMS
                        File containing Receptor atom types of your modelfile
                        (1 per line)
  -l LIGATOMS, --ligatoms LIGATOMS
                        File containing Ligand atom types of your modelfile (1
                        per line)
  -o OUTNAME, --outname OUTNAME
                        File containing commands to be evaluated to predict
                        grid points. Note: Requires GNINASCRIPTSDIR to be set
                        environment variable. (default:
                        simplegrid_predicts.txt)
  -t TYPESROOT, --typesroot TYPESROOT
                        Root folder for gninatypes data generated from script.
                        (default: simpletypes/)
  -m MODEL, --model MODEL
                        Model file that predictions will be made with. Must
                        end in .model
  -w WEIGHTS, --weights WEIGHTS
                        Weights for the model file that the predictions will
                        be made with.
  -n NUM_POINTS, --num_points NUM_POINTS
                        Number of points. Defaults are reasonable. (default:
                        200)
  -i INCREMENT, --increment INCREMENT
                        increment (Angstroms) between points. Combines with
                        num_points multiplicitavely. Defaults for both result
                        in 200 points spanning 20 angstroms (default: 0.1)
  -b BOX_SIZE, --box_size BOX_SIZE
                        Size of the box. Used for sanity check that you are
                        not trying to predict outside of box for gnina. MUST
                        MATCH BOX OF MODEL. Defaults are default grid size for
                        gnina (default: 24)
  --plot                Flag to make 1 large plot from the data. Assumes
                        job(s) have completed. Requires Hydrogen to be a vaild
                        receptor. Saves pdf called simple_vis.pdf in the
                        current working directory (default: False)
  -d DATAROOT, --dataroot DATAROOT
                        Root folder of data resulting from output of running
                        the OUTNAME file (default: simpledata/)
```

The workflow for this script is the following: 1) Generate OUTNAME, 2) run each command present in OUTNAME, 3) Plot

As an example, I will use the default values of the script, RECATOMS=my_recatoms.txt, LIGATOMS=my_ligatoms.txt, 
MODEL=my_model.model, and WEIGHTS=my_modelweights.caffemodel.

First, we need to generate the commands to run with gnina. (this is OUTNAME, which will be simplegrid_predicts.txt)
```
python simple_grid_visualization.py -m my_model.model -w my_modelweights.caffemodel -r my_recatoms.txt -l my_ligatoms.txt
```

This will generate 2 new directories: simpledata (empty) and simpletypes. Simpleteypes should have a directory for every
unique rec and lig atom. Additionally there will be a .types file for every rec+lig atom combination. Each types file 
should be NUM_POINTS lines (200 in this case).

Additionally, in the current working directory, there should be a new file called OUTNAME (simplegrid_predicts.txt).
It will have 1 line per rec and lig atom combination.

Now we need to set the GNINASCRIPTSDIR environment variable. This would correspond to where this repo is installed.
```
export GNINASCRIPTSDIR=~/git/gnina/scripts
```

Third, we need to evaluate each line in simplegrid_predicts.txt. Note: this could take a fair amount of time on one machine,
as this CANNOT be parallelized due to each job requiring the use of the GPU.
```
sh simplegrid_predicts.txt
```

After the above command has completed, simpledata should now be populated with <recatom>_rec_<ligatom>_lig_<modelname>_predictscores files. 
These files can be utilized to make graphs now.
```
python simple_grid_visualization.py -m my_model.model -w my_modelweights.caffemodel -r my_recatoms.txt -l my_ligatoms.txt --plot
```
This will make 1 large pdf containing all the plots, simple_vis.pdf.

### Case 2: Using grid_visualization.py
```
usage: grid_visualization.py [-h] -r RECATOMS -l LIGATOMS [-o OUTNAME]
                             [-t TYPESROOT] -m MODEL -w WEIGHTS [-p TEST_PDB]
                             [-c CUBE_LENGTH] [-n NUM_POINTS] [--make_dx]
                             [-d DATAROOT]

Script for generating the jobs that need to be run for visualization.
Generates types files & a text file that needs to be run. Can make a DX file
for visualization

optional arguments:
  -h, --help            show this help message and exit
  -r RECATOMS, --recatoms RECATOMS
                        File containing Receptor atom types of your modelfile (1 per line)
  -l LIGATOMS, --ligatoms LIGATOMS
                        File containing Ligand atom types of your modelfile (1 per line)
  -o OUTNAME, --outname OUTNAME
                        File containing commands to be evaluated to predict
                        grid points. Note: Requires GNINASCRIPTSDIR to be a
                        set environment variable. (default: grid_predicts.txt)
  -t TYPESROOT, --typesroot TYPESROOT
                        Root folder for gninatypes data generated from script.
                        (default: types/)
  -m MODEL, --model MODEL
                        Model file that predictions will be made with. Must
                        end in .model
  -w WEIGHTS, --weights WEIGHTS
                        Weights for the model file that the predictions will
                        be made with.
  -p TEST_PDB, --test_pdb TEST_PDB
                        pdbfile of receptor, centered at the origin for
                        visualization (default: gly_gly_gly.pdb)
  -c CUBE_LENGTH, --cube_length CUBE_LENGTH
                        Width of cube for grid box of points. Defaults are
                        reasonable (default: 24.0)
  -n NUM_POINTS, --num_points NUM_POINTS
                        Number of points per half of the box (ex 20 means
                        there will be 39x39x39 points total). Defaults are
                        reasonable. (default: 20)
  --make_dx             Flag to make dx files from the data. Assumes job(s)
                        have completed. (default: False)
  -d DATAROOT, --dataroot DATAROOT
                        Root folder of data resulting from output (default:
                        data/)
```
The first run of this script is WITHOUT the `--make_dx` flag. This will produce a file `OUTNAME`
wherin each line will need to be executed, and `GITSCRIPTSDIR` is a set environment variable which
points to the location where you have cloned this repository. The `gly_gly_gly.pdb` file is provided
in this directory, and is 3 connected glycine residues whose center of mass is at (0,0,0).

As before I will be evaluating the defaults of the script with RECATOMS=my_recatoms.txt, LIGATOMS=my_ligatoms.txt, 
MODEL=my_model.model, and WEIGHTS=my_modelweights.caffemodel. Note: gly_gly_gly.pdb needs to be in the current 
working directory, and the gninatyper tool needs to be installed on your machine (it is installed alongside gnina).

First we must make the textfile with all of the commands to be run with gnina.
```
python grid_visualization.py -m my_model.model -w my_modelweights.caffemodel -r my_recatoms.txt -l my_ligatoms.txt
```

Next, we need to execute each command in `OUTNAME` (grid_predicts.txt). NOTE: this cannot be parallelized on one machine
as each commands requires the GPU to be utilized. 
```
sh grid_predicts.txt
```

After all of the lines in `OUTNAME` have been completed, rerun this script with the `--make_dx` flag
and with the same arguments as before.
```
python grid_visualization.py -m my_model.model -w my_modelweights.caffemodel -r my_recatoms.txt -l my_ligatoms.txt --make_dx
```

`DATAROOT` will now contain the corresponding dx files. In order to visualize load the receptor and a dx file of interest
via pymol: `pymol TEST_PDB my_dxfile`. Visualizations can be performed by adjusting the volume slider & color of the dx object.
