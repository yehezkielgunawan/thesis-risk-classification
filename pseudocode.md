<!-- Pseudocode of scraping the dataset for word embedding purpose -->

Import the `Sort` and `reviews` modules from the `google_play_scraper` library.
Import the `pandas` library and alias it as `pd`.
Create an empty list called `reviewArray`.
Scrape reviews from the Bobobox app using the `reviews` function and store the result in `result1` and `continuation_token1`.
Extend the `reviewArray` list with the `result1` list.
Scrape reviews from the Traveloka app using the `reviews` function and store the result in `result2` and `continuation_token2`.
Extend the `reviewArray` list with the `result2` list.
Create a Pandas DataFrame called `feedbackPD` from the `reviewArray` list.
Select the `content` column of the `feedbackPD` DataFrame.

<!-- Pseudocode of converting the dataframe into excel file -->

Import `openpyxl` library.
Mount the Google Drive to the Colab notebook.
Set the path to the Excel file you want to export the DataFrame to.
Export the `content` column of the `feedbackPD` DataFrame to the Excel file using the `to_excel()` method.
Name the sheet of the Excel file as "Sheet1".

<!-- Pseudocode of developing the custom word embedding -->

Import the `openpyxl` library.
Import the `BertTokenizer` and `BertModel` modules from the `transformers` library.
Import the `torch` and `numpy` libraries.

Mount the Google Drive to the Colab notebook.
Set the path to the Excel file you want to export the DataFrame to.
Export the `content` column of the `feedbackPD` DataFrame to the Excel file using the `to_excel()` method.
Name the sheet of the Excel file as "Sheet1".

Set the `model_name` variable to 'bert-base-multilingual-cased'.
Create a BERT tokenizer using the `BertTokenizer.from_pretrained()` method and the specified `model_name`.
Create a BERT model using the `BertModel.from_pretrained()` method and the specified `model_name`.
Get the `content` column of the Pandas DataFrame `df` and convert it to a list called `sentences`.
Tokenize the `sentences` list using the `tokenizer()` method and convert them into input tensors.
Set the `device` variable to `"cuda"` if a GPU is available, otherwise set it to `"cpu"`.
Move the `model`, `input_ids`, and `attention_mask` tensors to the `device`.
Set the `batch_size` variable to 16.
Create an empty list called `embeddings`.
Loop through the `sentences` list in batches of size `batch_size`.
Get the batch input IDs and attention masks for the current batch and move them to the `device`.
Pass the batch input IDs and attention masks to the `model()` method to get the outputs.
Get the last hidden state of the outputs and select the first token of each sequence.
Convert the resulting tensor to a numpy array and append it to the `embeddings` list.
Stack the `embeddings` list into a numpy array.

<!-- Pseudocode of exporting the custom BERT pre-trained model -->

Set the `output_file` variable to "[file_path]".
Save the `embeddings` numpy array to the `output_file` using the `np.save()` method.

<!-- Pseudocode of Dataset Initialization in Scenario 1 -->

Import the `pandas`, `numpy`, `sklearn`, `matplotlib`, and `seaborn` libraries.

Load the new dataset for sentiment classification from the Excel file using the `pd.read_excel()` method and store it in the `new_data` variable.
Get the `content` column of the Pandas DataFrame `new_data` and convert it to a list called `sentences`.
Load the exported word embeddings from the file "[file_path]" using the `np.load()` method and store it in the `embeddings` variable.

<!-- Pseudocode of Dataset Initialization in Scenario 2 -->

Import the `pandas`, `numpy`, `sklearn`, `matplotlib`, and `seaborn` libraries.
Import the `torch` and `transformers` libraries.

Load the new dataset for sentiment classification from the Excel file using the `pd.read_excel()` method and store it in the `new_data` variable.

Create an IndoBERT tokenizer using the `AutoTokenizer.from_pretrained()` method and the specified model name.
Create an IndoBERT model using the `AutoModel.from_pretrained()` method and the specified model name.
Tokenize the `sentences` list using the `tokenizer()` method and convert them into input tensors.
Pass the input tensors to the `model()` method to get the outputs.
Get the last hidden state of the outputs and store it in the `embeddings` variable.
Convert the `embeddings` tensor to a numpy array.

<!-- Pseudocodes of Random Forest Implementation on each scenarios -->

Import the sklearn library.
Import the RandomForestClassifier, GridSearchCV, confusion_matrix, and classification_report modules.
Import the numpy library.

Define the param_grid1 dictionary with the hyperparameters for priority_score.
Define the param_grid2 dictionary with the hyperparameters for problem_domain.

Create two RandomForestClassifier objects with the class_weight parameter set to "balanced" and the random_state parameter set to 42.

Create two GridSearchCV objects with the estimator parameter
Set to the corresponding RandomForestClassifier object and the param_grid parameter
Set to the corresponding param_grid dictionary.
Set the cv parameter to 5 and fit the objects to the training data.

Get the best classifiers with the optimal hyperparameters by calling the best estimator attribute of the GridSearchCV objects.

<!-- Pseudocodes of Predicting the test dataset using Random Forest Model -->

Predict the labels of the test sets using the best classifiers and store the results in y_pred1 and y_pred2.

Generate the classification reports for the test sets using the classification_report() method
Store the results in classification_rep1 and classification_rep2.

Print the best hyperparameters and classification reports for both priority_score and problem_domain.
