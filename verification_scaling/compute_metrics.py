import argparse
from collections import Counter
import datasets

def compute_accuracy(rewards, gt_rewards):
    """
    Compute accuracy of the rewards prediction.

    Args:
        rewards: List of lists, where each element is a list of 0s and 1s
        gt_rewards: List of lists, where each element is a list of 0s and 1s (ground truth)

    Returns:
        float: Accuracy of the predictions
    """
    correct_ones = 0

    for reward_seq, gt_reward_seq in zip(rewards, gt_rewards):
        # Ensure both sequences have the same length
        assert len(reward_seq) == len(gt_reward_seq), "Sequence lengths don't match"

        for i, reward in enumerate(reward_seq):
            if reward == 1:
                if gt_reward_seq[i] == 1:
                    correct_ones += 1
                break

    return correct_ones / len(rewards)

def compute_scores(actual, predicted):
    """
    Compute F1 score between two lists of binary values (0s and 1s).
    
    Parameters:
    actual (list): List of actual values (ground truth)
    predicted (list): List of predicted values
    
    Returns:
    float: F1 score
    """
    # Validate inputs
    if len(actual) != len(predicted):
        raise ValueError("Input lists must have the same length")
    
    # Count true positives, false positives, and false negatives
    true_positives = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    false_positives = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 1)
    false_negatives = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 0)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positives_rate = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    false_negatives_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return f1, false_positives_rate, false_negatives_rate

def main():
    parser = argparse.ArgumentParser(description="Compute accuracy of reward predictions")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--rewards_column", type=str, default="rewards", 
                      help="Column name for predicted rewards")
    parser.add_argument("--gt_rewards_column", type=str, default="gt_rewards", 
                      help="Column name for ground truth rewards")
    args = parser.parse_args()
    
    # Load the dataset
    print(f"Loading dataset {args.dataset}, split {args.split}...")
    dataset = datasets.load_dataset(args.dataset, split=args.split)
    
    # Extract the columns
    rewards = dataset[args.rewards_column]
    gt_rewards = dataset[args.gt_rewards_column]
    test_cases = [example['verification_info']['test_cases'] for example in dataset]

    print("average #test cases:", sum(len(i) for i in test_cases) / len(test_cases))
    
    # Compute the accuracy
    print("Computing accuracy...")
    accuracy = compute_accuracy(rewards, gt_rewards)
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    num_generations = len(dataset[args.rewards_column][0])
    num_passes = sum(1 for one in gt_rewards if sum(one) > 0)
    pass_at_k = num_passes / len(gt_rewards)
    print(f"pass@{num_generations}: {pass_at_k} ({pass_at_k*100:.2f}%)")

    flat_rewards = []
    flat_gt_rewards = []
    for example in dataset:
        if len(example['verification_info']['test_cases']) == 0:
            flat_rewards += [-1] * len(example[args.rewards_column])
            flat_gt_rewards += example[args.gt_rewards_column]
        else:
            flat_rewards += example[args.rewards_column]
            flat_gt_rewards += example[args.gt_rewards_column]
    test_gen_ha_accuracy = [i==j for i, j in zip(flat_rewards, flat_gt_rewards)]
    test_gen_ha_accuracy = sum(test_gen_ha_accuracy) / len(test_gen_ha_accuracy)
    print(f"test_gen_ha_accuracy: {test_gen_ha_accuracy} ({test_gen_ha_accuracy*100:.2f}%)")
    test_gen_f1, test_gen_false_positives_rate, test_gen_false_negatives_rate = compute_scores(flat_gt_rewards, flat_rewards)
    print(f"test_gen_f1: {test_gen_f1} ({test_gen_f1*100:.2f}%)")
    print(f"test_gen_false_positives_rate: {test_gen_false_positives_rate} ({test_gen_false_positives_rate*100:.2f}%)")
    print(f"test_gen_false_negatives_rate: {test_gen_false_negatives_rate} ({test_gen_false_negatives_rate*100:.2f}%)")
    test_gen_accuracy = [i==j for i, j in zip(rewards, gt_rewards)]
    test_gen_accuracy = sum(test_gen_accuracy) / len(test_gen_accuracy)
    print(f"test_gen_accuracy: {test_gen_accuracy} ({test_gen_accuracy*100:.2f}%)")
    malformed_rate = Counter(flat_rewards)[-1] / len(flat_rewards)
    print(f"malformated test cases: {malformed_rate} ({malformed_rate*100:.2f}%)")

    filtered_rewards = []
    filtered_gt_rewards = []
    for i, j in zip(rewards, gt_rewards):
        if 0 in j and 1 in j:
            filtered_rewards.append(i)
            filtered_gt_rewards.append(j)
    test_gen_accuracy = [i==j for i, j in zip(filtered_rewards, filtered_gt_rewards)]
    test_gen_accuracy = sum(test_gen_accuracy) / len(test_gen_accuracy)
    print(f"filtered test_gen_accuracy (total of {len(filtered_rewards)}): {test_gen_accuracy} ({test_gen_accuracy*100:.2f}%)")
if __name__ == "__main__":
    main()
