from data_processor import DATA_PROCESSORS
from tasks import TASKS
import json
import random

OOD_REVIEWS_SUBSET = [
    'all',
    'Chicago',
    'non-Chicago'
]

"""
Rosa: commented out these two functions (get_test_data, get_train_data) because using get_data will be more replicable.
"""
# def get_test_data(args):
#     print('get test data')
#     _, test_data = get_data(args.task, 0, args.num_test)
#     return test_data

# def get_train_data(args):
#     print('get train data')
#     train_data, _ = get_data(args.task, args.num_train, 0)
#     return train_data

def get_data(args):
    task_name = args.task
    num_train = args.num_train
    num_test = args.num_test
    num_val = args.num_val

    print('task_name:', task_name)
    task = TASKS[task_name]()
    if task_name == 'shoe':
        data_processor = DATA_PROCESSORS[task_name](task.train_data_path, num_train, num_test)

        train_data = data_processor.get_train_data()
        test_data = data_processor.get_test_data()
        val_data = None

    elif task_name in ['hotel_reviews', 'headline_binary', 'retweet']:
        train_data_processor = DATA_PROCESSORS[task_name](task.train_data_path, num_train, is_train=True)
        test_data_processor = DATA_PROCESSORS[task_name](task.test_data_path, num_test)
        val_data_processor = DATA_PROCESSORS[task_name](task.val_data_path, num_val)

        train_data = train_data_processor.get_data()
        test_data = test_data_processor.get_data()
        val_data = val_data_processor.get_data()

    else:
        raise ValueError('task_name defined:', task_name)
    
    if task_name == 'hotel_reviews' and args.use_ood_reviews in OOD_REVIEWS_SUBSET:
        print(f"Loading {args.use_ood_reviews} OOD hotel reviews.")
        with open(f'{task.ood_test_data_path}/ood_hotel_reviews_{args.use_ood_reviews}.json', 'r') as file:
            ood_data = json.load(file)

        ood_reviews = ood_data['review']
        ood_labels = ood_data['label']
        random.seed(49)
        num_samples = min(num_test,len(ood_labels))
        reviews, labels = zip(*random.sample(list(zip(ood_reviews, ood_labels)), num_samples))
        test_data = {
            'review': reviews,
            'label': labels
        }
    
    if task_name != 'hotel_reviews' and args.use_ood_reviews in OOD_REVIEWS_SUBSET:
        raise ValueError("Only hotel reviews dataset has OOD samples.")
        

    return train_data, test_data, val_data
