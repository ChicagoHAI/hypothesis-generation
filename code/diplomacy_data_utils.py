import pandas as pd
from utils import get_dialogue_id

def change_country_names(df, output_path):
    """
    Star-Warization
    We are suspicious of data leakage problem, since the Diplomacy dataset is published before GPT-3.5. Thus, we swap the country names in Diplomacy with Star War characters' names.
    """
    countries2character = {
        "Germany": "Darth Vader",
        "Italy": "Yoda",
        "Russia": "Luke Skywalker",
        "Turkey": "Han Solo",
        "Austria": "R2-D2",
        "England": "Obi-Wan Kenobi",
        "France": "Chewbacca",
        # and the lower case versions
        "germany": "Darth Vader",
        "italy": "Yoda",
        "russia": "Luke Skywalker",
        "turkey": "Han Solo",
        "austria": "R2-D2",
        "england": "Obi-Wan Kenobi",
        "france": "Chewbacca",
        # and the adjectives
        "German": "Darth Vader",
        "Italian": "Yoda",
        "Russian": "Luke Skywalker",
        "Turkish": "Han Solo",
        "Austrian": "R2-D2",
        "English": "Obi-Wan Kenobi",
        "French": "Chewbacca",
        # and the lower case versions
        "german": "Darth Vader",
        "italian": "Yoda",
        "russian": "Luke Skywalker",
        "turkish": "Han Solo",
        "austrian": "R2-D2",
        "english": "Obi-Wan Kenobi",
        "french": "Chewbacca",
        # and the adjectives
        "Germany's": "Darth Vader's",
        "Italy's": "Yoda's",
        "Russia's": "Luke Skywalker's",
        "Turkey's": "Han Solo's",
        "Austria's": "R2-D2's",
        "England's": "Obi-Wan Kenobi's",
        "France's": "Chewbacca's",
        # and the lower case versions
        "germany's": "Darth Vader's",
        "italy's": "Yoda's",
        "russia's": "Luke Skywalker's",
        "turkey's": "Han Solo's",
        "austria's": "R2-D2's",
        "england's": "Obi-Wan Kenobi's",
        "france's": "Chewbacca's",
    }

    df = df.replace(countries2character)
    for k,v in countries2character.items():
        def replace_k_with_v(lst):
            return [x.replace(k, v) if isinstance(x, str) else x for x in lst]
        df = df.applymap(lambda x: replace_k_with_v(x) if isinstance(x, list) else x)
        df = df.applymap(lambda x: x.replace(k, v) if isinstance(x, str) else x)
        
    print(df.head())
    df.to_json(output_path, orient='records', lines=True)

def game_df_operations(game_path):
    game_df = pd.read_json(game_path, lines=True)

    # get rid of index = 9 for all columns (because it is empty, but this is *only for game 1*)
    if game_df.messages[9] == []:
        game_df = game_df.drop([9])
        game_df = game_df.reset_index(drop=True)

    # set dialogue_id to be the index
    if 'dialogue_id' not in game_df.columns.tolist():
        game_df['dialogue_id'] = game_df.index

    # get number of messages per dialogue
    if 'num_messages' not in game_df.columns.tolist():
        game_df['num_messages'] = game_df.messages.apply(lambda x: len(x))

    # update game_1_data
    game_df.to_json(game_path, orient='records', lines=True)

def split_truthful_and_deceptive_messages(game_df, game_id):
    # Put all truthful messages into a dictionary
    # Put all deceptive messages into a dictionary
    truthful_data = {}
    deceptive_data = {}
    for idx in range(len(game_df.messages.explode())):
        # get information
        messages = game_df.messages.explode().to_list()[idx]
        sender_labels = game_df.sender_labels.explode().to_list()[idx]
        receiver_label = game_df.receiver_labels.explode().to_list()[idx]
        speaker = game_df.speakers.explode().to_list()[idx]
        receiver = game_df.receivers.explode().to_list()[idx]
        absolute_message_index = game_df.absolute_message_index.explode().to_list()[idx]
        relative_message_index = game_df.relative_message_index.explode().to_list()[idx]
        season = game_df.seasons.explode().to_list()[idx]
        year = game_df.years.explode().to_list()[idx]
        game_score = game_df.game_score.explode().to_list()[idx]
        game_score_delta = game_df.game_score_delta.explode().to_list()[idx]

        game_id = game_id
        num_messages = game_df.num_messages.explode().to_list()
        dialogue_id = get_dialogue_id(num_messages, idx)

        # add to dataframe
        if sender_labels == True:
            truthful_data['messages'] = [messages] if 'messages' not in truthful_data else truthful_data['messages'] + [messages]
            truthful_data['sender_labels'] = [sender_labels] if 'sender_labels' not in truthful_data else truthful_data['sender_labels'] + [sender_labels]
            truthful_data['receiver_labels'] = [receiver_label] if 'receiver_labels' not in truthful_data else truthful_data['receiver_labels'] + [receiver_label]
            truthful_data['speakers'] = [speaker] if 'speakers' not in truthful_data else truthful_data['speakers'] + [speaker]
            truthful_data['receivers'] = [receiver] if 'receivers' not in truthful_data else truthful_data['receivers'] + [receiver]
            truthful_data['absolute_message_index'] = [absolute_message_index] if 'absolute_message_index' not in truthful_data else truthful_data['absolute_message_index'] + [absolute_message_index]
            truthful_data['relative_message_index'] = [relative_message_index] if 'relative_message_index' not in truthful_data else truthful_data['relative_message_index'] + [relative_message_index]
            truthful_data['seasons'] = [season] if 'seasons' not in truthful_data else truthful_data['seasons'] + [season]
            truthful_data['years'] = [year] if 'years' not in truthful_data else truthful_data['years'] + [year]
            truthful_data['game_score'] = [game_score] if 'game_score' not in truthful_data else truthful_data['game_score'] + [game_score]
            truthful_data['game_score_delta'] = [game_score_delta] if 'game_score_delta' not in truthful_data else truthful_data['game_score_delta'] + [game_score_delta]
            truthful_data['game_id'] = [game_id] if 'game_id' not in truthful_data else truthful_data['game_id'] + [game_id]
            truthful_data['dialogue_id'] = [dialogue_id] if 'dialogue_id' not in truthful_data else truthful_data['dialogue_id'] + [dialogue_id]
        elif sender_labels == False:
            deceptive_data['messages'] = [messages] if 'messages' not in deceptive_data else deceptive_data['messages'] + [messages]
            deceptive_data['sender_labels'] = [sender_labels] if 'sender_labels' not in deceptive_data else deceptive_data['sender_labels'] + [sender_labels]
            deceptive_data['receiver_labels'] = [receiver_label] if 'receiver_labels' not in deceptive_data else deceptive_data['receiver_labels'] + [receiver_label]
            deceptive_data['speakers'] = [speaker] if 'speakers' not in deceptive_data else deceptive_data['speakers'] + [speaker]
            deceptive_data['receivers'] = [receiver] if 'receivers' not in deceptive_data else deceptive_data['receivers'] + [receiver]
            deceptive_data['absolute_message_index'] = [absolute_message_index] if 'absolute_message_index' not in deceptive_data else deceptive_data['absolute_message_index'] + [absolute_message_index]
            deceptive_data['relative_message_index'] = [relative_message_index] if 'relative_message_index' not in deceptive_data else deceptive_data['relative_message_index'] + [relative_message_index]
            deceptive_data['seasons'] = [season] if 'seasons' not in deceptive_data else deceptive_data['seasons'] + [season]
            deceptive_data['years'] = [year] if 'years' not in deceptive_data else deceptive_data['years'] + [year]
            deceptive_data['game_score'] = [game_score] if 'game_score' not in deceptive_data else deceptive_data['game_score'] + [game_score]
            deceptive_data['game_score_delta'] = [game_score_delta] if 'game_score_delta' not in deceptive_data else deceptive_data['game_score_delta'] + [game_score_delta]
            deceptive_data['game_id'] = [game_id] if 'game_id' not in deceptive_data else deceptive_data['game_id'] + [game_id]
            deceptive_data['dialogue_id'] = [dialogue_id] if 'dialogue_id' not in deceptive_data else deceptive_data['dialogue_id'] + [dialogue_id]

    truthful_df = pd.DataFrame(truthful_data)
    deceptive_df = pd.DataFrame(deceptive_data)

    return truthful_df, deceptive_df

def read_data_by_labels(path, game_id):
    """
    Get the distribution of labels in the dataset
    """
    game_df = pd.read_json(path, lines=True)
    truthful_df, deceptive_df = split_truthful_and_deceptive_messages(game_df, game_id)

    # Get label class distribution
    print(f"Number of truthful messages: {game_df.sender_labels.explode().sum()}")
    print(f"Number of deceptive messages: {len(game_df.sender_labels.explode()) - game_df.sender_labels.explode().sum()}")

    print(f'Percent of deceptive messages: {round((len(game_df.sender_labels.explode()) - game_df.sender_labels.explode().sum()) / len(game_df.sender_labels.explode()) * 100, 2)}%')

    return truthful_df, deceptive_df

def sample_messages(truthful_df, deceptive_df, n_truthful, n_deceptive, seed=42):
    truthful_df_sampled = truthful_df.sample(n=n_truthful, random_state=seed)
    deceptive_df_sampled = deceptive_df.sample(n=n_deceptive, random_state=seed)

    return truthful_df_sampled, deceptive_df_sampled

def combine_labels_and_save(truthful, deceptive, path):
    # combine the two dfs
    df = pd.concat([truthful, deceptive])
    # save to output
    df.to_json(path, orient='records', lines=True)

def sample_and_save(directory, num_pos, num_neg, game_id, task):
    input_path = f'{directory}/{task}_game_id_{game_id}.jsonl'
    print('input_path: ', input_path)
    truthful_df, deceptive_df = read_data_by_labels(input_path, game_id=game_id)

    # Sample truthful and deceptive messages
    pos_sampled, neg_sampled = sample_messages(truthful_df, deceptive_df, 90, 10)

    # combine the two dfs and save to output
    output_path = f'{directory}/{task}_game_id_{game_id}_{num_pos}_{num_neg}_class.jsonl'
    print('output_path: ', output_path)
    combine_labels_and_save(pos_sampled, neg_sampled, output_path)