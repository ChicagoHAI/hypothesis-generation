from tasks import BaseTask
import json
import random

OOD_REVIEWS_SUBSET = [
    'all',
    'Chicago',
    'non-Chicago'
]


def get_data(args):
    task_name = args.task
    num_train = args.num_train
    num_test = args.num_test
    num_val = args.num_val

    print('task_name:', task_name)
    try:
        task = BaseTask(task_name)

        train_data, test_data, val_data = task.get_data(num_train, num_test, num_val)
    except FileNotFoundError:
        raise ValueError('task_name undefined:', task_name)

    if task_name == 'hotel_reviews' and args.use_ood_reviews in OOD_REVIEWS_SUBSET:
        print(f"Loading {args.use_ood_reviews} OOD hotel reviews.")
        with open(f'{task.ood_test_data_path}/ood_hotel_reviews_{args.use_ood_reviews}.json', 'r') as file:
            ood_data = json.load(file)

        ood_reviews = ood_data['review']
        ood_labels = ood_data['label']
        random.seed(49)
        num_samples = min(num_test, len(ood_labels))
        reviews, labels = zip(*random.sample(list(zip(ood_reviews, ood_labels)), num_samples))
        test_data = {
            'review': reviews,
            'label': labels
        }

    if task_name != 'hotel_reviews' and args.use_ood_reviews in OOD_REVIEWS_SUBSET:
        raise ValueError("Only hotel reviews dataset has OOD samples.")

    return train_data, test_data, val_data
