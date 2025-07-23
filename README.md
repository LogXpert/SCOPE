# SCOPE

SCOPE: a Tree-based Self-Correcting Online Log Parsing mehtod with Syntactic-Semantic Collaboration.
It introduces a novel bi-directional tree structure that enables efficient template matching from both forward and reverse directions, resulting in a higher overall matching rate. Additionally, it adopts a two-stage syntactic-semantic collaboration framework: a lightweight NLP model first utilizes POS information for syntax-based matching, while the LLM is selectively invoked as a fallback to handle semantically complex cases when uncertainty remains. This design significantly reduces LLM API usage while maintaining high accuracy, achieving a balance between efficiency and effectiveness.

SW framework is extented based on Drain3.


## Directory Structure

- `code/`: Core modules for masking, profiling, and template mining
- `evaluator/`: Evaluation scripts, configuration, and datasets
- `examples/`: Example scripts and configuration files

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/LogXpert/SCOPE.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run example scripts:
   ```bash
   python examples/test.py

4. Run dataset evaluation:
   ```bash
   python evaluator/evaluator.py


## Result

Result can be found in result.log under running directory

Bi-Tree:True, Pool:True, POS:True, LLM:True, model:Qwen/Qwen3-235B-A22B, thinking:True
|  # | Dataset     |   GA     |    PA    |   Dur(sec) |   #Line |   #Template | #Upd(Fwd) | #Upd(Rev) | #Upd(Pool) | #LLM Call | #LLM Tokens |
|----|-------------|----------|----------|------------|---------|-------------|-----------|-----------|------------|-----------|-------------|
|  0 | HPC         | 0.99     | 0.991    |    330.25  |    2000 |      49     |       3   |    2      |      1     |    8      |     17430   |
|  1 | OpenStack   | 1        | 0.939    |     96.17  |    2000 |      43     |       0   |    0      |      0     |    2      |      3922   |
|  2 | BGL         | 0.9935   | 0.9475   |   1092.88  |    2000 |     118     |       0   |    1      |      2     |   19      |     39998   |
|  3 | HDFS        | 1        | 1        |     17.39  |    2000 |      14     |       0   |    0      |      0     |    0      |         0   |
|  4 | Hadoop      | 0.9935   | 0.8955   |    336.15  |    2000 |     115     |       0   |    0      |      1     |    6      |     12645   |
|  5 | Spark       | 0.999    | 0.996    |    261.73  |    2000 |      35     |       0   |    0      |      1     |    6      |     12761   |
|  6 | Zookeeper   | 0.9945   | 0.974    |    300.42  |    2000 |      57     |       1   |    0      |      0     |    7      |     14638   |
|  7 | Thunderbird | 0.9575   | 0.7805   |   1425.72  |    2000 |     201     |       4   |    3      |      4     |   30      |     63371   |
|  8 | Windows     | 1        | 0.978    |    222.27  |    2000 |      50     |       0   |    0      |      1     |    4      |      8578   |
|  9 | Linux       | 0.998    | 0.98     |   1291.68  |    2000 |     117     |       2   |    1      |      0     |   23      |     47111   |
| 10 | Andriod     | 0.9595   | 0.683    |    750.84  |    2000 |     163     |       5   |    5      |      1     |   13      |     30531   |
| 11 | HealthApp   | 1        | 0.8645   |     78.71  |    2000 |      75     |       0   |    0      |      0     |    2      |      4143   |
| 12 | Apache      | 1        | 1        |     16.59  |    2000 |       6     |       0   |    0      |      0     |    0      |         0   |
| 13 | Proxifier   | 1        | 1        |     57.69  |    2000 |       8     |       0   |    0      |      0     |    1      |      2069   |
| 14 | OpenSSH     | 1        | 0.9965   |    676.97  |    2000 |      26     |       2   |    1      |      3     |    9      |     20699   |
| 15 | Mac         | 0.9655   | 0.6595   |   2163.78  |    2000 |     345     |       7   |    2      |      0     |   27      |     60401   |
| 16 | Average     | 0.990688 | 0.917812 |   569.953  |    2000 |      88.875 |       1.5 |    0.9375 |      0.875 |    9.8125 |     21143.6 |

## Usage

- Configure log parsing and template mining using `.ini` files in `evaluator/`.
- Use provided datasets for benchmarking and evaluation.
- Extend core modules in `code/` for custom log formats.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and new features.

## License

This project is open-source. See the LICENSE file for details. 
