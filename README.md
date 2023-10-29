# growhonsup
Python implementation of algorithm for generating a Higher-Order Network (HON) from sequence input in a supervised setting. 

This project is essentially an extension of [GrowHON](https://github.com/sjkrieg) for working with supervised data. It accepts any text file as input, and processes each line as a sequence vector. The output is a weighted adjacency list in CSV format. The output graph for this project differs from other HON models in that it creates additional sink nodes for each of the outcomes.

### Core dependencies
1. Python 3.7+
2. numpy (tested on 1.23.1)
3. pandas (tested on 1.4.3)
4. tqdm (tested on 4.65.0)

The key difference between this project and prior HON models is the use of an outcome variable to determine which conditional (higher-order) nodes to create in the graph. These outcomes must be supplied as part of the input. Each line in the input file should follow this template:
```
SEQUENCE_ID 1 2 3 ... n LABEL
```
where SEQUENCE_ID is an integer ID for the sequence, 1..n are elements in the sequence, and LABEL={0.,1.} is a string representing either a negative outcome (0.) or positive outcome (1.) associated with that sequence. Please see the repository for an example input file.

### Required (positional) Arguments:
```
  infname               source path + file name
  otfname               destination path + file name
  k                     max order to use in growing the HON
```

### Optional Arguments:
```
  -h, --help            show this help message and exit
  -w NUMCPUS, --numcpus NUMCPUS
                        number of workers for multiprocessing (integer; default 16)
  -x MINSUP, --minsup MINSUP
                        minimum frequency for a higher-order pattern to be considered (integer, default 100)
  -z Z, --z Z
                        significance threshold for information gain (IG) computations (float, default 10.0)
  -di INFDELIMITER, --infdelimiter INFDELIMITER
                        delimiter for entities in input vectors (char; default " ")
                        this is the character by which GrowHON delimits entities in each input vector
  -do OTFDELIMITER, --otfdelimiter OTFDELIMITER
                        delimiter for output network (char; default " ")
  -o LOGNAME, --logname LOGNAME
                        location to write log output (string; default None)
                        if None, all log messages are printed to console
  --noskip NOSKIP
                        do not allow skip connections in the graph (bool, default False)
  --saveigs SAVEIGS
                        location to save information gains as .csv file (str, default None)
```

### Example
Create a HON using default parameters.
```
python growhonsup.py sample_sequences.txt sample_hon_k3.txt 3
```

### Example
Use fewer worker processes and decrease the significance threshold to result in a larger graph.
```
python growhonsup.py sample_sequences.txt sample_hon_k3.txt 3 -w 4 -z 3.0
```

## Authors

* **Steven Krieg** - (skrieg@nd.edu)

## Reference
If you make use of this code, please cite our paper!

* Paper ref TBD
