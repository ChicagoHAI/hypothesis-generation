import pandas as pd 

file_path = 'headline.csv'
df = pd.read_csv(file_path)

df.sort_values(by='clickability_test_id', inplace=True)
df.drop_duplicates(subset=['clickability_test_id', 'headline'], keep=False, inplace=True)




df['clickthrough_rate'] = df['clicks'] / df['impressions']

df['headline'] = df['headline'].str.strip('\'"')

option_one_df = df[['clickability_test_id', 'headline', 'clicks']]
unique_test_ids = option_one_df['clickability_test_id'].unique()

for i, test_ids in enumerate(unique_test_ids):
        filtered_rows = option_one_df[option_one_df['clickability_test_id'] == test_ids]
        sorted_rows = filtered_rows.sort_values(by='clicks', axis=0, ascending=False)
        #print(sorted_rows)
        if i != 0:
            sorted_rows.to_csv('headline_grouped_dataset.csv', mode='a', index=False, header=False)
        else:
            sorted_rows.to_csv('headline_grouped_dataset.csv')

for i, test_ids in enumerate(unique_test_ids):

    filtered_rows = option_one_df[option_one_df['clickability_test_id'] == test_ids]
    filtered_rows.drop_duplicates(subset=['headline'], keep=False)
    sorted_rows = filtered_rows.sort_values(by='clicks', axis=0, ascending=False)
    if len(sorted_rows) > 1:
        if i != 0:
            sorted_rows.iloc[0:1].to_csv('headline_binary_dataset.csv', mode='a', index=False, header=False) 
            sorted_rows.iloc[-1:].to_csv('headline_binary_dataset.csv', mode='a', index=False, header=False)

        else:
            sorted_rows.iloc[0:1].to_csv('headline_binary_dataset.csv', index=False)
            sorted_rows.iloc[-1:].to_csv('headline_binary_dataset.csv', mode='a', index=False, header=False)

option_2_df = df[['clickability_test_id', 'headline', 'clickthrough_rate']]
sorted_rows = option_2_df.sort_values(by='clickthrough_rate', axis=0, ascending=False)
sorted_rows.to_csv('headline_fully_sorted.csv')
