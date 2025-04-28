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
    test_gen_accuracy = [i==j for i, j in zip(flat_rewards, flat_gt_rewards)]
    test_gen_accuracy = sum(test_gen_accuracy) / len(test_gen_accuracy)
    print(f"test_gen_accuracy: {test_gen_accuracy} ({test_gen_accuracy*100:.2f}%)")
    malformed_rate = Counter(flat_rewards)[-1] / len(flat_rewards)
    print(f"malformated test cases: {malformed_rate} ({malformed_rate*100:.2f}%)")

if __name__ == "__main__":
    main()
