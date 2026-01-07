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

import re
import os
import sys
import json
import time
import tempfile
import subprocess
import math
from typing import Dict, List, Tuple, Optional, Any

def extract_python_code(model_output: str) -> Optional[str]:
    """Extract Python code from model output using regex"""
    # Look for Python code blocks
    python_patterns = [
        r'```python\s*\n(.*?)\n```',
        r'```gurobi\s*\n(.*?)\n```',
        r'```py\s*\n(.*?)\n```',
    ]
    
    for pattern in python_patterns:
        matches = re.findall(pattern, model_output, re.DOTALL)
        if matches:
            # Return the last (most complete) Python block
            code = matches[-1].strip()
            return _ensure_function_call(code)
    
    # If the string itself looks like code (no markdown blocks), try it directly
    if "from gurobipy import" in model_output or "import gurobipy" in model_output:
            return _ensure_function_call(model_output.strip())

    return None

def _ensure_function_call(code: str) -> str:
    """
    Ensure that if code defines a function, it also calls it.
    """
    # Check if gurobipy is installed
    check_import = """
try:
    import gurobipy
except ImportError:
    print("CRITICAL_ERROR: gurobipy is not installed")
    exit(1)
"""
    code = check_import + code
    
    # Check if there's already a function call or main guard
    if 'if __name__ == "__main__":' in code or '__main__' in code:
        return code
    
    # Look for the specific solve_problem definition we expect
    function_match = re.search(r'def\s+(\w+)\([^)]*\):', code)
    if function_match:
        function_name = function_match.group(1)
        # Check if function is called outside of def
        # Simple check: count occurrences of "function_name("
        # If only 1 (the def), then it needs a call.
        if code.count(f"{function_name}(") <= 1:
                return f"{code}\n\n# Auto-generated function call\nif __name__ == \"__main__\":\n    {function_name}()"
    
    return code

def _prepare_code_for_execution(python_code: str) -> str:
    """
    Modify code to capture the returned model object and print its status/results
    in a format we can parse.
    """
    # Find where the function is called
    if 'if __name__ == "__main__":' in python_code:
        # We will append our own extraction logic instead of trusting the user's print statements
        # But we need to capture the return value of the function call.
        # Strategy: Replace the main block with our own.
        base_code = python_code.split('if __name__ == "__main__":')[0]
        
        # Find the function name again to call it
        function_match = re.search(r'def\s+(\w+)\([^)]*\):', base_code)
        if function_match:
            func_name = function_match.group(1)
            
            extraction_code = f"""
if __name__ == "__main__":
    try:
        model = {func_name}()
        
        if model:
            print(f"_MODEL_STATUS_: {{model.status}}")
            
            if model.status == 2:  # GRB.OPTIMAL
                print(f"_MODEL_OBJ_: {{model.objVal}}")
                for v in model.getVars():
                    try:
                        print(f"_MODEL_VAR_ {{v.VarName}} = {{v.X}}")
                    except:
                        pass
            else:
                print("_MODEL_OBJ_: NA")
        else:
                print("ERROR: Function did not return a model object")
                
    except Exception as e:
        print(f"ERROR: {{e}}")
"""
            return base_code + extraction_code

    return python_code

def _parse_output(stdout: str) -> Tuple[Optional[int], Optional[float], Dict[str, float]]:
    """Parse the standardized output format"""
    status = None
    obj_value = None
    var_values = {}
    
    lines = stdout.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith("_MODEL_STATUS_:"):
            status_str = line.split(":", 1)[1].strip()
            try:
                status = int(status_str)
            except ValueError:
                status = None
        elif line.startswith("_MODEL_OBJ_:"):
            obj_str = line.split(":", 1)[1].strip()
            if obj_str != "NA":
                try:
                    obj_value = float(obj_str)
                except ValueError:
                    pass
        elif line.startswith("_MODEL_VAR_"):
            # Format: _MODEL_VAR_ x0 = 1.23
            parts = line.replace("_MODEL_VAR_", "").split("=", 1)
            if len(parts) == 2:
                var_name = parts[0].strip()
                try:
                    var_value = float(parts[1].strip())
                    var_values[var_name] = var_value
                except ValueError:
                    pass
    
    return status, obj_value, var_values

def execute_code_safely(python_code: str, time_limit: int = 10) -> Tuple[bool, Optional[int], Optional[float], Dict[str, float]]:
    """Execute Python code in a sandboxed environment"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare code
        modified_code = _prepare_code_for_execution(python_code)
        
        code_file = os.path.join(temp_dir, "code_exec.py")
        with open(code_file, 'w') as f:
            f.write(modified_code)
        
        env = os.environ.copy()
        
        try:
            result = subprocess.run(
                [sys.executable, code_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=time_limit,
                env=env
            )
            
            exec_success = result.returncode == 0
            
            # Raise error if gurobipy is missing to crash the training
            if "CRITICAL_ERROR: gurobipy is not installed" in result.stdout:
                raise ImportError("CRITICAL: gurobipy is not installed in the environment! Training cannot proceed.")

            status, obj_value, var_values = _parse_output(result.stdout)
            
            return exec_success, status, obj_value, var_values
            
        except subprocess.TimeoutExpired:
            return False, None, None, {}
        # except Exception:
        #     return False, None, None, {}

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    Args:
        solution_str: The LLM generated solution
        ground_truth: JSON string containing the problem metadata and correct solution info
        method: Not used, kept for API compatibility
        format_score: Score if code executes but result is wrong
        score: Score if result is correct
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        try:
            gt_data = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Fallback if it's not a valid json string
            return 0.0
    else:
        gt_data = ground_truth

    # Extract code
    python_code = extract_python_code(solution_str)
    if not python_code:
        print("unable to extract")
        return 0.0

    # Execute
    exec_success, status, obj_value, var_values = execute_code_safely(python_code)

    
    if not exec_success:
        print("unable to exec")
        return 0.0

    # Validate against Ground Truth
    meta = gt_data.get("meta", {})
    result_block = gt_data.get("gurobi_result", {})
    
    ref_optimum = result_block.get("theoretical_optimum")
    if ref_optimum is None:
        ref_optimum = meta.get("theoretical_optimum", 0.0)

    ref_vars = result_block.get("optimal_values")
    if ref_vars is None:
        ref_vars = meta.get("optimal_values", {})

    # 1. Check Status
    ref_status = result_block.get("solver_status", meta.get("solver_status"))
    # Gurobi Optimal is 2
    if ref_status is None: 
        ref_status = 2 
    
    status_correct = (status == ref_status)
    
    if not status_correct:
        print("status is wrong")
        return format_score # Executed but wrong status (e.g. infeasible)

    print("generated:", obj_value, "ground truth:", ref_optimum)

    # 2. Check Objective
    tolerance_obj = 1e-4
    objective_matches = False
    if obj_value is not None:
        if abs(obj_value - ref_optimum) < tolerance_obj:
            objective_matches = True
            
    if not objective_matches:
        return format_score

    # 3. Check Variables
    tolerance_var = 1e-4
    
    ref_values = sorted(list(ref_vars.values()))
    gen_values = sorted(list(var_values.values()))

    print("generated:", gen_values, "ground truth:", ref_values)

    vars_match = False
    if not ref_values and not gen_values:
        vars_match = True
    elif not ref_values or not gen_values:
        vars_match = False
    else:
        if len(ref_values) == len(gen_values):
            matches = []
            for r, g in zip(ref_values, gen_values):
                matches.append(abs(r - g) < tolerance_var)
            vars_match = all(matches)
    
    if vars_match:
        return score
    else:
        return format_score

