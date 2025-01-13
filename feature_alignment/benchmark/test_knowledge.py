def test_knowledge(results, json_path):
    total_clean_acc = 0
    total_corrupt_acc_a = 0 
    total_corrupt_acc_b = 0
    
    for i, result in enumerate(results):
        clean_answer_count = 0
        corrupted_answer_count_a = 0
        corrupted_answer_count_b = 0
        useless_count = 0
        for each_result in result:
            clean_answer = each_result["clean_answer"]
            corrupted_answer = each_result["corrupted_answer"]
            if clean_answer == corrupted_answer:
                useless_count += 1
            if clean_answer.lower() in each_result["clean_output"].lower():
                clean_answer_count += 1
            if corrupted_answer.lower() in each_result["corrupted_output"].lower():
                corrupted_answer_count_b += 1
            if clean_answer.lower() in each_result["corrupted_output"].lower():
                corrupted_answer_count_a += 1
        
        clean_acc = clean_answer_count / (len(result) - useless_count)
        corrupt_acc_a = corrupted_answer_count_a / (len(result) - useless_count)
        corrupt_acc_b = corrupted_answer_count_b / (len(result) - useless_count)
        
        total_clean_acc += clean_acc
        total_corrupt_acc_a += corrupt_acc_a
        total_corrupt_acc_b += corrupt_acc_b
        
        print(f"Layer {i}:")
        print(f"Clean answer count: {clean_answer_count}")
        print(f"Corrupted answer count A: {corrupted_answer_count_a}")
        print(f"Corrupted answer count B: {corrupted_answer_count_b}")
        print(f"Useless count: {useless_count}")
        print(f"Accuracy: {clean_acc}")
        print(f"Corrupted answer accuracy A: {corrupt_acc_a}")
        print(f"Corrupted answer accuracy B: {corrupt_acc_b}")
    
    num_layers = len(results)
    print("\nAverages across all layers:")
    print(f"Average clean accuracy: {total_clean_acc / num_layers:.3f}")
    print(f"Average corrupted accuracy A: {total_corrupt_acc_a / num_layers:.3f}")
    print(f"Average corrupted accuracy B: {total_corrupt_acc_b / num_layers:.3f}")