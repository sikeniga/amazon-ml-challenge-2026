import sys
import subprocess

# Run a dry-run test of generate_final_submission by temporarily checking syntax and function execution
print("Testing generate_final_submission.py dry-run...")
from generate_final_submission import clean_name, squish, clean_addr, get_street_num, get_street_prefix, get_city_tokens, compute_features
f = compute_features("apple inc", "appleinc", "123 main st", "123", "US", "apple corporation", "applecorporation", "123 main street", "123", "US", 100)
print(f"Features compute test: len={len(f)}, values={f}")
assert len(f) == 11, "Feature length mismatch"
print("All dry-run assertions passed successfully!")
