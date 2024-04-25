 import pandas as pd
import numpy as np
np.random.seed(0)
clothing_id = np.random.randint(1000, 2000, 1000)  
age = np.random.randint(18, 65, 1000)  
review = ["Great product!", "Not satisfied with the quality.", "Perfect fit and comfortable.", "Disappointed with the color."] * 250 
rating = np.random.randint(1, 6, 1000)  
recommended = np.random.choice([True, False], 1000)  
feedback = ["Love it!", "Could be better.", "Highly recommend!", "Not worth the price."] * 250  
category = np.random.choice(['Tops', 'Dresses', 'Pants', 'Skirts'], 1000)  
data = pd.DataFrame({
    'Clothing_ID': clothing_id,
    'Age': age,
    'Review': review,
    'Rating': rating,
    'Recommended': recommended,
    'Feedback': feedback,
    'Category': category
})
data.to_csv('clothing_reviews_dataset.csv', index=False)


#code for creating data set 

import pandas as pd
data = {
    'Clothing_ID': [1001, 1002, 1003, 1004, 1005],
    'Age': [25, 30, 35, 40, 45],
    'Review': ["Great product!", "Not satisfied with the quality.", "Perfect fit and comfortable.", "Disappointed with the color.", "Love it!"],
    'Rating': [5, 2, 5, 3, 5],
    'Recommended': [True, False, True, False, True],
    'Feedback': ["Love it!", "Could be better.", "Highly recommend!", "Not worth the price.", "Excellent service!"],
    'Category': ['Tops', 'Dresses', 'Pants', 'Skirts', 'Tops']
}
df = pd.DataFrame(data)
print(df)
df.to_csv('clothing_reviews_dataset.csv', index=False)
