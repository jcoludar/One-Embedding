# Integrated Branch
## DATA
To ensure same data

https://syncandshare.lrz.de/getlink/fiNWgfGhmaCgpmbhsj3eUR/

unfiltered.json is needed in trizod dict. Refer to config.py
## Main scripts

### run_v2_validation_all_data.py
All datasets with flexibility.
Saves after each test. Except baseline, which is recalculated each run. Can resume on same json if crash or terminated.

Would not recommend running all at once, can batch them in separate runs, save load support on single json. 

Buckets can be precalculated and saved together in one json per dataset, look at BUCKETS dict in config.py for reference. only supports per prot buckets, but cant imagine implementing per residue being too dificult. Turn buckets lists into dicts, saving index to keep, and then check in bucket script if instanceof(dict) and change clusters for each bucket. bucket_eval() in helpers/loaders_runners.py

This is very simplified and missing special features made by everyone in group. Simplified to match original structure with added flexibility. 

--tests 

list of datasets to run on

--out_json_name

output name simple. Is placed in folder designated by config.py

--input_config

replaces V2_CONFIGS with file based config selection

```bash
python run_v2_validation_all_data.py --tests ts115 casp12 cb513 chezod trizod scope scope_superfamily cath20 deeploc --out_json_name d_out_sweep.json --input_config input_configs/d_out_sweep.json
```

### pytorch_v3_validation.py
per residue only. Prot not impossible to implement.
Very Simple pytorch model. One hidden layer, dropout, batchnorm2d, relu.

Made separate csv format for some reason. Consitent csvs between all datasets I guess. Process outlined in per_res_to_single_csv.py script

Saves after each test.
Uses first input of config as baseline, was too lazy to make reader for raw h5. just use lossless 1024. separate configs from /input_configs/.

encodes all h5s as first step.

Parameter tuning is done using. 
```python
nominal_grid = { "lr": [0.05,0.0005],"wd": [0.0005],"dropout": [0.1, 0.3] }
```
program doesnt do param tuning on multiple seeds as it takes too long. Only like 3 epochs per test in param tuning and then 5 during final model.

Disorder changed average cluster to median cluster, sometimes model breaks and has negative rho, which messed up average. Rare and made graphs look chaotic every 10 or so points.  

--tests 

list of datasets to run on

--out_json_name

output name simple. Is placed in folder designated by config.py

--input_config

replaces V2_CONFIGS with file based config selection
```bash
python pytorch_v3_validation.py --tests chezod trizod cb513 casp12 ts115 -o pytorch_d_out_sweep_context.json -c input_configs/d_out_sweep_pytorch_context.json
```
