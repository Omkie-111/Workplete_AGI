# AI-Powered Assistant Functions Documentation

This GitHub documentation provides an overview and usage instructions for a Python script containing various functions that leverage different AI models for different tasks, such as filtering houses, generating Salesforce prompts, logging calls, and extracting actor ages from Wikipedia. The script uses libraries like `openai`, `pandas`, `torch`, `transformers`, and `simple_salesforce` to achieve these tasks.

## Table of Contents
- [Introduction](#introduction)
- [Function List](#function-list)
  - [`house_prompt(prompt)`](#house_promptprompt)
  - [`sf_prompt(prompt)`](#sf_promptprompt)
  - [`log_prompt(prompt)`](#log_promptprompt)
  - [`age_prompt(prompt)`](#age_promptprompt)
  - [`process_prompt(prompt)`](#process_promptprompt)
- [Usage Examples](#usage-examples)
- [Requirements](#requirements)

## Introduction

This Python script contains functions that serve as an AI-powered assistant to perform various tasks using different AI models. These tasks include finding houses, generating Salesforce prompts, logging calls, and extracting actor ages from Wikipedia. The script utilizes a combination of libraries and APIs to execute these tasks efficiently.

## Function List

### `house_prompt(prompt)`

This function searches for houses in Houston based on user criteria provided in the prompt.

**Parameters:**
- `prompt` (str): A user prompt containing information about the desired house.

**Usage Example:**
```python
result = house_prompt("Find me a house in Houston that works for 4. My budget is 600k")
print(result)
```

### `sf_prompt(prompt)`
This function generates a Salesforce prompt for adding a new lead using the information provided in the prompt.

**Parameters:**
- `prompt` (str): A user prompt containing information for adding a new lead.

**Usage Example:**
```python
result = sf_prompt("Add Max Nye at Workplete as the new lead.")
print(result)
```

### `log_prompt(prompt)`
This function generates a text using an AI assistant and logs a call on Salesforce based on the generated text.

**Parameters:**
- `prompt` (str): A user prompt for generating a call log.

**Usage Example:**
```python
result = log_prompt("Log a call with James Veel.")
print(result)
```

### `age_prompt(prompt)`
This function uses a text-to-text generation model to extract the age of an actor from a Wikipedia page based on the prompt.

**Parameters:**
- `prompt` (str): A user prompt containing information about the actor.

**Usage Example:**
```python
result = age_prompt("How old is the actor in Mace Windu on https://www.wikipedia.org/")
print(result)
```

### `process_prompt(prompt)`
This function processes different prompts and routes them to the appropriate sub-functions based on keywords in the prompt.

**Parameters:**
- `prompt` (str): A user prompt to be processed.

**Usage Example:**
```python
# House Prompt Example
result = house_prompt("Find me a house in Houston that works for 4. My budget is 600k")
print(result)

# Salesforce Prompt Example
result = sf_prompt("Add Max Nye at Workplete as the new lead.")
print(result)

# Log Prompt Example
result = log_prompt("Log a call with James Veel.")
print(result)

# Actor Age Prompt Example
result = age_prompt("How old is the actor in Mace Windu on https://www.wikipedia.org/")
print(result)

# Process Prompt Example
result = process_prompt("Find me a house in Houston that works for 4. My budget is 600k")
print(result)
```

