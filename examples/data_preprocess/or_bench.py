# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the OR-LLM-Synthetic-Bench dataset to parquet format
"""

import argparse
import os
import json
import glob
import textwrap
import pandas as pd
from pathlib import Path

from verl.utils.hdfs_io import copy, makedirs

PROMPT_TEMPLATE = textwrap.dedent(
    """
    Solve the following optimization problem using Gurobi. 
    
    {LLM_DESCRIPTION}

    Write a Python function `solve_problem()` using `gurobipy` to solve this.

    Requirements:
    1. The solution MUST be a valid Python code block wrapped in ```python ... ```.
    2. The function `solve_problem` must return the Gurobi model object `m` after calling `m.optimize()`.
    3. Define variables with the EXACT names and types (Continuous/Integer) as specified in the problem.
    4. Do not test the function or print solutions; just provide the function definition.
    5. Do not import any libraries other than `gurobipy`.
    6. Ensure the code is self-contained and clear, with comments explaining the logic.

    Template:
    ```python
    from gurobipy import *

    def solve_problem():
        m = Model("Optimization_Problem")
        # Define variables, objective, and constraints
        # ...
        m.optimize()
        return m
    ```
    """
).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--raw_dataset_path", default="synthetic_dataset/resource_allocation", help="The local path to the raw dataset directory.")
    parser.add_argument(
        "--local_save_dir", default="~/data/or_bench", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    
    raw_dataset_path = args.raw_dataset_path
    
    # Check if raw dataset path exists
    if not os.path.exists(raw_dataset_path):
        raise FileNotFoundError(f"Raw dataset path {raw_dataset_path} not found.")

    data_source = "or_bench"

    def process_split(split_name, folder_name):
        split_dir = os.path.join(raw_dataset_path, folder_name)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist. Skipping {split_name} split.")
            return []

        files = glob.glob(os.path.join(split_dir, "*.json"))
        print(f"Found {len(files)} files in {split_dir}")
        
        dataset_rows = []
        
        for file_path in sorted(files):
            try:
                with open(file_path, 'r') as f:
                    problem_data = json.load(f)
                
                llm_description = problem_data.get("LLM_description")
                if not llm_description:
                    print(f"Skipping {file_path}: Missing LLM_description")
                    continue
                    
                prompt = PROMPT_TEMPLATE.format(LLM_DESCRIPTION=llm_description)
                
                # Construct the ground truth object for the reward model
                # We need all the validation info
                ground_truth = {
                    "meta": problem_data.get("meta", {}),
                    "gurobi_result": problem_data.get("gurobi_result", {}),
                    "variables": problem_data.get("variables", {}),
                    "constraints": problem_data.get("constraints", {}),
                    "gold_solution": problem_data.get("gold_solution", "")
                }
                
                # JSON serialize ground_truth because it needs to be a string in some contexts, 
                # but verl usually expects a string or simple object. 
                # However, the reward function will receive this.
                # In verl gsm8k example, ground_truth is a string (the answer).
                # Here we pass a json string to be parsed by the reward function.
                ground_truth_str = json.dumps(ground_truth)

                row = {
                    "data_source": data_source,
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "ability": "optimization",
                    "reward_model": {"style": "rule", "ground_truth": ground_truth_str},
                    "extra_info": {
                        "split": split_name,
                        "problem_id": os.path.basename(file_path),
                        "original_file": file_path
                    },
                }
                dataset_rows.append(row)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return dataset_rows

    train_data = process_split("train", "train")
    test_data = process_split("test", "eval")
    train_2_11_data = process_split("train_2_11", "train_2_11")

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    
    # Expand user path
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    if train_data:
        train_df = pd.DataFrame(train_data)
        train_df.to_parquet(os.path.join(local_save_dir, "train.parquet"))
        print(f"Saved {len(train_df)} training examples to {os.path.join(local_save_dir, 'train.parquet')}")

    if test_data:
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_save_dir, "test.parquet"))
        print(f"Saved {len(test_df)} test examples to {os.path.join(local_save_dir, 'test.parquet')}")
    
    if train_2_11_data:
        train_2_11_df = pd.DataFrame(train_2_11_data)
        train_2_11_df.to_parquet(os.path.join(local_save_dir, "train_2_11.parquet"))
        print(f"Saved {len(train_2_11_df)} train_2_11 examples to {os.path.join(local_save_dir, 'train_2_11.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
