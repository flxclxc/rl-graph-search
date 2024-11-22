# Define the sets
set1=("facebook_414" "facebook_686" "facebook_348" "facebook_0" "facebook_3437")
# set1 = ("ablation_0", "ablation_1", "ablation_2", "ablation_3", "ablation_4", "ablation_5")

# Iterate over the sets
for entry1 in "${set1[@]}"; do
    python src/test_baselines.py --graph "${entry1}"
    python src/test_on_all_seeds.py --graph $entry1
    done