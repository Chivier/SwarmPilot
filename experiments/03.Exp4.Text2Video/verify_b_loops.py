import numpy as np

def verify_sampling(num_workflows_to_generate):
    b_loop_rng = np.random.default_rng(42)
    
    first_half_loops = []
    second_half_loops = []
    
    print(f"Simulating {num_workflows_to_generate} workflows...")
    
    for i in range(num_workflows_to_generate):
        if i < num_workflows_to_generate / 2:
            max_b_loops = int(b_loop_rng.integers(1, 4))  # Random 1-3
            first_half_loops.append(max_b_loops)
        else:
            max_b_loops = int(b_loop_rng.integers(3, 6))  # Random 3-5
            second_half_loops.append(max_b_loops)
            
    print("\nResults:")
    print(f"First half (1-3 expected): Min={min(first_half_loops)}, Max={max(first_half_loops)}")
    print(f"Second half (3-5 expected): Min={min(second_half_loops)}, Max={max(second_half_loops)}")
    
    # Check if values are within expected ranges
    assert all(1 <= x <= 3 for x in first_half_loops), "First half values out of range!"
    assert all(3 <= x <= 5 for x in second_half_loops), "Second half values out of range!"
    
    print("\nVerification PASSED!")

if __name__ == "__main__":
    verify_sampling(100)
