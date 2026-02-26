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

def _check_reference_constraints(var_values: Dict[str, float], ref_vars_info: Dict, constraints: Dict, tol: float = 1e-4) -> float:
    """
    Check what fraction of the reference constraints are satisfied by the generated solution.

    For each reference constraint Ci with sense and rhs, computes:
        LHS = sum over Var_j of (resource_costs[Var_j][Ci] * var_values[Var_j])
    and checks whether LHS sense rhs within tolerance.

    Returns fraction of constraints satisfied (0.0 to 1.0). Returns 0.0 if no named
    variables from the reference appear in var_values (can't evaluate).
    """
    if not constraints or not var_values or not ref_vars_info:
        return 0.0

    # Only proceed if at least one reference variable name is present in generated output
    matching_vars = [v for v in ref_vars_info if v in var_values]
    if not matching_vars:
        return 0.0

    satisfied = 0
    total = len(constraints)

    for cname, cdata in constraints.items():
        sense = cdata.get("sense")
        rhs = cdata.get("rhs")
        if sense is None or rhs is None:
            total -= 1
            continue

        lhs = 0.0
        for vname, vinfo in ref_vars_info.items():
            coeff = vinfo.get("resource_costs", {}).get(cname, 0.0)
            val = var_values.get(vname, 0.0)
            lhs += coeff * val

        if sense == ">=" and lhs >= rhs - tol:
            satisfied += 1
        elif sense == "<=" and lhs <= rhs + tol:
            satisfied += 1
        elif sense == "=" and abs(lhs - rhs) <= tol:
            satisfied += 1

    if total <= 0:
        return 0.0
    return satisfied / total


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0, alpha=10.0):
    """
    Hierarchical + continuous partial-credit reward for OR-Bench optimization problems.

    Scoring breakdown (sums to 1.0 for a perfect solution):
      +0.05  code extracted successfully
      +0.10  code executes without error
      +0.10  solver reaches OPTIMAL status (or +0.05 for feasible-not-optimal)
      +0.05  generated variable count matches reference
      +0.20  fraction of reference constraints satisfied by generated solution (continuous)
      +0.20  objective closeness: exp(-alpha * rel_gap), only when OPTIMAL (continuous)
      → 1.0  override: exact objective AND exact sorted variable values (within 1e-4)

    Args:
        solution_str: The LLM generated solution
        ground_truth: JSON string (or dict) with problem metadata and correct solution info
        method: Not used, kept for API compatibility
        format_score: Legacy parameter, no longer used but kept for API compatibility
        score: Legacy parameter for exact-match score ceiling (default 1.0)
        alpha: Decay rate for the objective-closeness exponential (default 10.0).
               Higher alpha = faster decay away from the optimum.
    """
    # Parse ground truth
    if isinstance(ground_truth, str):
        try:
            gt_data = json.loads(ground_truth)
        except json.JSONDecodeError:
            return 0.0
    else:
        gt_data = ground_truth

    # ------------------------------------------------------------------
    # GATE 1: Code extraction (+0.05)
    # ------------------------------------------------------------------
    python_code = extract_python_code(solution_str)
    if not python_code:
        print("unable to extract")
        return 0.0

    reward = 0.05

    # ------------------------------------------------------------------
    # GATE 2: Execution (+0.10)
    # ------------------------------------------------------------------
    exec_success, status, obj_value, var_values = execute_code_safely(python_code)

    if not exec_success:
        print("unable to exec")
        return reward  # 0.05

    reward += 0.10  # 0.15

    # Resolve ground-truth reference values
    meta = gt_data.get("meta", {})
    result_block = gt_data.get("gurobi_result", {})

    ref_optimum = result_block.get("theoretical_optimum")
    if ref_optimum is None:
        ref_optimum = meta.get("theoretical_optimum", 0.0)

    ref_vars = result_block.get("optimal_values") or meta.get("optimal_values", {})

    ref_status = result_block.get("solver_status") or meta.get("solver_status") or 2

    # ------------------------------------------------------------------
    # GATE 3: Solver status (+0.10 optimal, +0.05 feasible-not-optimal)
    # ------------------------------------------------------------------
    GUROBI_OPTIMAL = 2
    GUROBI_SUBOPTIMAL_FEASIBLE = {5, 13}  # SUBOPTIMAL, SOLUTION_LIMIT

    status_is_optimal = (status == ref_status == GUROBI_OPTIMAL)
    status_is_feasible = (status in GUROBI_SUBOPTIMAL_FEASIBLE)

    if status_is_optimal:
        reward += 0.10  # 0.25
    elif status_is_feasible:
        reward += 0.05  # 0.20 — feasible but not optimal
    else:
        print(f"status is wrong: generated={status}, expected={ref_status}")
        return reward  # 0.15

    # ------------------------------------------------------------------
    # GATE 4: Variable count match (+0.05)
    # ------------------------------------------------------------------
    if ref_vars and var_values and len(var_values) == len(ref_vars):
        reward += 0.05  # up to 0.30

    # ------------------------------------------------------------------
    # CONTINUOUS A: Reference constraint satisfaction (+0.20)
    # Uses named variable values and reference constraint/coefficient data.
    # ------------------------------------------------------------------
    ref_vars_info = gt_data.get("variables", {})
    constraints = gt_data.get("constraints", {})

    constraint_fraction = _check_reference_constraints(var_values, ref_vars_info, constraints)
    reward += 0.20 * constraint_fraction
    print(f"constraint satisfaction: {constraint_fraction:.3f}")

    # ------------------------------------------------------------------
    # CONTINUOUS B: Objective closeness (+0.20, only when OPTIMAL)
    # exp(-alpha * rel_gap): 1.0 at exact match, decays smoothly with error
    # ------------------------------------------------------------------
    if status_is_optimal and obj_value is not None and ref_optimum is not None:
        print(f"generated obj: {obj_value}, ground truth: {ref_optimum}")
        rel_gap = abs(obj_value - ref_optimum) / (abs(ref_optimum) + 1e-6)
        obj_score = math.exp(-alpha * rel_gap)
        reward += 0.20 * obj_score
        print(f"objective closeness: {obj_score:.3f} (rel_gap={rel_gap:.4f})")

    # ------------------------------------------------------------------
    # OVERRIDE: Exact solution → 1.0
    # Objective within 1e-4 AND all sorted variable values within 1e-4
    # ------------------------------------------------------------------
    tolerance = 1e-4

    objective_exact = (
        obj_value is not None
        and ref_optimum is not None
        and abs(obj_value - ref_optimum) < tolerance
    )

    if objective_exact and ref_vars and var_values:
        ref_values = sorted(ref_vars.values())
        gen_values = sorted(var_values.values())
        print(f"generated vars: {gen_values}, ground truth: {ref_values}")

        if len(ref_values) == len(gen_values):
            if all(abs(r - g) < tolerance for r, g in zip(ref_values, gen_values)):
                return score  # 1.0

    return reward

