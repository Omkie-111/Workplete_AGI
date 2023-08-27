import openai
import pandas as pd
import torch
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
from simple_salesforce import Salesforce
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Create openai API key
openai.api_key = "add_openai_apikey"

# Authenticate to Salesforce API
salesforce_secrets = "add_salesforce_secret_key_here"

def house_prompt(prompt):
    # dataset
    data = pd.read_csv('houston_house.csv')

    df = data[["ADDRESS","PROPERTY TYPE","CITY","STATE OR PROVINCE","ZIP OR POSTAL CODE","PRICE","BEDS","BATHS","LOCATION","SQUARE FEET","LOT SIZE","DAYS ON MARKET","$/SQUARE FEET","STATUS"]]

    df.rename(columns = {"ADDRESS" : "address","PROPERTY TYPE" : "property_type","CITY" : "city","STATE OR PROVINCE" : "state","ZIP OR POSTAL CODE" : "zipcode","PRICE" : "price","BEDS" : "beds","BATHS" : "baths","LOCATION" : "location","SQUARE FEET" : "sqft","LOT SIZE" : "lot_size","DAYS ON MARKET" : "days_on_market","$/SQUARE FEET" : "$/sqft","STATUS" : "status"}, inplace = True)

    # Create a pipeline for the DistilBERT model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)
    nlp = pipeline('text-classification', model=model, tokenizer=tokenizer)

    # Define function to filter houses based on criteria
    def filter_houses(df, criteria):
        filtered_df = df.copy()
        for key, value in criteria.items():
            if key == "price":
                filtered_df = filtered_df[filtered_df["price"] <= value]
            else:
                filtered_df = filtered_df[filtered_df[key] >= value]
        return filtered_df
        
        
    # Define function to generate response based on user input
    def generate_response(prompt):
        # Define criteria based on user input
        criteria = {}
        if "Houston" in prompt:
            criteria["city"] = "Houston"
        if "works for 4" in prompt:
            criteria["beds"] = 4
        if "budget is" in prompt:
            criteria["price"] = int(prompt.split('budget is ')[1].replace('k', '')) * 1000

        # Filter houses based on criteria
        filtered_df = filter_houses(df, criteria)

        # Generate response
        if filtered_df.empty:
            response = "I'm sorry, I couldn't find any houses that match your criteria."
        else:
            house = filtered_df.sample().iloc[0]
            address = house["address"]
            price = house["price"]
            response = f"I found a house at {address} for {price}. Would you like more information?"

        return response

    # Define function to get user input and generate response
    def chat(prompt):
        response = generate_response(prompt)
        return response
    
    return chat(prompt)

# Test the AGI assistant function
prompt = "Find me a house in Houston that works for 4. My budget is 600k"

def sf_prompt(prompt):
    
    salesforce_username = salesforce_secrets["username"]
    salesforce_password = salesforce_secrets["password"]
    salesforce_security_token = salesforce_secrets["security_token"]

    # Define the prompt
    prompt = (f"Add Max Nye at Workplete as the new lead.\n"
            f"Salesforce username: {salesforce_username}\n"
            f"Salesforce password: {salesforce_password}\n"
            f"Salesforce security token: {salesforce_security_token}")

    # Initialize the DistilBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    # Generate the input_ids and attention_mask for the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Use the model to predict the sentiment of the prompt
    outputs = model(input_ids, attention_mask=attention_mask)
    predicted_class = outputs.logits.argmax().item()

    # Convert the predicted class to a string label
    labels = ["negative", "positive"]
    predicted_label = labels[predicted_class]

    # Print the predicted sentiment label
    return "The sentiment of the prompt is " + predicted_label

def log_prompt(prompt):

    sf = Salesforce(username = salesforce_secrets["username"],
                    password = salesforce_secrets["password"],
                    security_token = salesforce_secrets["security_token"])

    # create function to log a call on Salesforce
    def log_call(name, message):
        sf.Lead.create({"LastName": name})
        sf.Task.create({"WhoId": sf.Lead.get_by_name(name)['Id'],
                        "Subject": "Phone Call",
                        "Status": "Completed",
                        "Description": message})

    # set up transformer pipeline
    nlp = pipeline("text-generation", 
                model="distilgpt2", 
                tokenizer="distilgpt2", 
                device=0 if torch.cuda.is_available() else -1)

    # generate text with the AGI assistant
    output_text = nlp(prompt, max_length=1000, do_sample=True, temperature=0.7)[0]["generated_text"]

    # extract relevant information from the output text
    name = "James Veel"
    message = output_text.split("saying that ")[-1].split(".")[0]

    # log the call on Salesforce
    return log_call(name, message)

def age_prompt(prompt):
    # Define transformers pipeline
    nlp = pipeline("text2text-generation", model="EleutherAI/gpt-neo-2.7B")

    # Define function to generate response from prompt
    def generate_response(prompt):
        response = nlp(prompt, max_length=512)[0]['generated_text']
        return response.strip()
    
    # Look up actor on Wikipedia
    response = generate_response(prompt + " on https://www.wikipedia.org/")
    
    # Extract age from response
    age = None
    for word in response.split():
        if word.isdigit():
            age = int(word)
            break
    
    # Return age or error message
    if age:
        return f"The actor is {age} years old."
    else:
        return "Could not determine age from response."

# Define function to process each prompt
def process_prompt(prompt):
    if "Find me a house in Houston" in prompt:
        return house_prompt(prompt)
    elif "Add Max Nye at Workplete" in prompt:
        # Code to add lead to Salesforce goes here
        return sf_prompt(prompt)
    elif "Log a call with James Veel" in prompt:
        # Code to log call in Salesforce goes here
        return log_prompt(prompt)
    elif "How old is the actor in Mace Windu" in prompt:
        # Code to log call in Salesforce goes here
        return age_prompt(prompt)
    else:
        return "Invalid prompt."