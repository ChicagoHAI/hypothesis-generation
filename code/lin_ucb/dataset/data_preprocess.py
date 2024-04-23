import pickle, numpy as np

file_path = "/data/rosa-shared/shoe_rec_1100.pkl"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

def create_data():
    with open("shoe_data", 'w') as f:
        for x in data:
            f.write(str(x)+"\n")


def extract_info(text):
    
    # Process the input text
    
    # Initialize variables to store gender, descriptors, colors, and bag descriptor
    gender = None
    descriptor1 = None
    descriptor2 = None
    colors = []
    bag_descriptor = None
    
    # Iterate through the tokens in the processed text
    x = text.split()
    for i, token in enumerate(text.split()):
        # Check for gender-related words
        if token.lower() in ['man', 'woman']:
            gender = token.lower()
        
        # Check for age-related words
        elif token.lower() in ['young', 'old']:
            descriptor1 = token.lower()
        
        # Check for height-related words
        elif token.lower() in ['tall', 'short']:
            descriptor2 = token.lower()
        
        # Check for color-related words
        elif token.lower() in ['white', 'red', 'blue', 'black', 'orange', 'green']:
            colors.append(token.lower())
        
        # Check for the word "bag" and extract the descriptor two words before it
        elif token.lower() == 'bag' and i >= 2:
            bag_descriptor = x[i - 2].lower()
    
    return gender, descriptor1, descriptor2, colors, bag_descriptor

def featurize(gender, descriptor1, descriptor2, colors, bag_descriptor):
    y = []
    if descriptor1 == 'young':
        y.append(0)
    else:
        y.append(1)
    
    if descriptor2 == 'tall':
        y.append(0)
    else:
        y.append(1)
    
    if gender == 'man':
        y.append(0)
    else:
        y.append(1)
    
    if bag_descriptor == "large":
        y.append(0)
    else:
        y.append(1)

    for i, color in enumerate(colors):   
        x = [0 for i in range(6)]
        if color == 'white':
            x[0] = 1
        if color == 'red':
            x[1] = 1
        if color == 'blue':
            x[2] = 1
        if color == 'black':
            x[3] = 1
        if color == 'orange':
            x[4] = 1
        if color == 'green':
            x[5] = 1
        if i != len(colors)-1:
            y = y + x
        else:
            y.append(np.argmax(np.array(x)))
    
    return y

# Example usage
with open("featurized", "w") as f:
    for x in data:
        gender, descriptor1, descriptor2, colors, bag_descriptor = extract_info(x[0])
        _, _, _, chosen_color, _ = extract_info(x[1])
        f.write(str(featurize(gender, descriptor1, descriptor2, colors+chosen_color, bag_descriptor)) + '\n')

        

