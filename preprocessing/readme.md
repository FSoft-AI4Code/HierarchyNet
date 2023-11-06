### Data processing
***
#### Our required data format
1. As our code supports distributed training, to increase the effencicy, we preprocess data to help distributed training run smoothly. The entire process is written in the *pipeline.py*. To use it, you need to prepare training/testing/validation files in the jsonline where each line is a json containing 2 fields: *code (input)* and *comment (output)*.
2. File *pipeline.py* receives 7 arguments:
    * train-path: the path of the training jsonline file
    * val-path: the path of the validation jsonline file
    * test-path: the path of the testing jsonline file
    * tokenizer-path: the folder contains created code/text tokenizers
    * lang-path: the path of the tree-sitter object
    * lang: programming language name in line with tree-sitter
    * output-path: the output folder contains the created files for the following steps
For example,
```
    python pipeline.py --train-path [train_path] \
                       --val-path [val_path] \
                       --test-path [test_path] \
                       --tokenizer-path [tokenizer_path] \
                       --lang-path [lang_path] \
                       --lang java \
                       --output-path [output_path]
```
3. Expected output folder has the structure as below. Necessary data of each sample is saved in a sub-folder (0,1,...) containing 8 files
```
<output_path>
    train
        0
            flattened_nodes.jsonl
            stmts.jsonl
            ast_node_ids.txt
            ast_dfs_ids.txt
            stmt_indices.txt
            G.pickle
            target.json
            metadata.json
        1
            ...
        ...
    test
        ...
    val
        ...
```

