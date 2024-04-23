# generates random predictions
import random
from utils import compute_binary_metrics
from data_loader import get_test_data
from tasks import TASKS


def generate_pred(seed, num_test_examples, label_classes):
    random.seed(seed)
    predictions = []
    for _ in range(num_test_examples):
        predictions.append(random.choice(label_classes))
    return predictions


def main():
    random.seed(42)
    seeds = random.sample(range(1000), 3)

    num_test_examples = -1
    task = 'diplomacy'
    assert task in ['diplomacy', 'nli']
    label_classes = TASKS[task]().label_classes

    f1_list = []
    acc_list = []
    for seed in seeds:
        # load labels
        test_data = get_test_data(num_test_examples, task)
        labels = test_data['label']

        preds = generate_pred(seed, num_test_examples if num_test_examples!=-1 else len(labels), label_classes)

        # compute metrics
        # print('labels: ', labels)
        # print('preds: ', preds)
        metrics = compute_binary_metrics(task, labels, preds)
        f1_list.append(metrics['f1'])
        acc_list.append(metrics['acc'])

    print("F1 Mean: ", sum(f1_list)/len(f1_list))
    print("F1 Std: ", sum([(f1 - sum(f1_list)/len(f1_list))**2 for f1 in f1_list])/(len(f1_list)-1))
    print("F1 Max: ", max(f1_list))
    print("F1 Min: ", min(f1_list))

    print("Acc Mean: ", sum(acc_list)/len(acc_list))
    print("Acc Std: ", sum([(acc - sum(acc_list)/len(acc_list))**2 for acc in acc_list])/(len(acc_list)-1))
    print("Acc Max: ", max(acc_list))
    print("Acc Min: ", min(acc_list))


if __name__=="__main__":
    main()