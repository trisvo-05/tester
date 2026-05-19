import os

# Danh sach 5 cau hinh Ablation
# Format: Ten_Config, n_slices, k_val, use_csca, weight_coh, weight_lat
ablations = [
    ("A1_no_CSCA",   3, 1, 0, 1.0, 1.0),
    ("A2_no_Lcoh",   3, 1, 1, 0.0, 1.0),
    ("A3_no_Llat",   3, 1, 1, 1.0, 0.0),
    ("A4_Sweep_K2",  3, 2, 1, 1.0, 1.0),
    ("A4_Sweep_K3",  3, 3, 1, 1.0, 1.0),
    ("A5_Sweep_N5",  5, 1, 1, 1.0, 1.0),
    ("A5_Sweep_N7",  7, 1, 1, 1.0, 1.0),
]

print("BAT DAU ABLATION SUITE (Tu dong hoa hoan toan)\n")

for config in ablations:
    name, n, k, csca, w_coh, w_lat = config
    cmd = (
        f"accelerate launch --num_processes 1 train_ablation.py "
        f"--config_name {name} "
        f"--n_slices {n} "
        f"--k_val {k} "
        f"--use_csca {csca} "
        f"--weight_coh {w_coh} "
        f"--weight_lat {w_lat} "
        f"--seed 42"
    )
    print(f"Dang thuc thi lenh: {cmd}")
    os.system(cmd)

print("\nDA XONG TOAN BO ABLATION SUITE! Hay mo file 'ablation_results.csv' de xem ket qua.")