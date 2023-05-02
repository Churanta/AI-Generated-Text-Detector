import nltk
from nltk.tokenize import word_tokenize
from transformers import pipeline, set_seed

nltk.download('punkt')

def detect_ai_text(text):
    # define the AI detection pipeline
    generator = pipeline('text-generation', model='gpt2', device=-1)
    # set a random seed for the model to get consistent results
    set_seed(42)
    # generate a sample of text using the pipeline
    generated_text = generator('', max_length=100, do_sample=True)[0]['generated_text']
    # tokenize the generated text and the input text
    gen_tokens = word_tokenize(generated_text)
    text_tokens = word_tokenize(text)
    # calculate the percentage of AI-related tokens in the input text
    ai_tokens = [t for t in text_tokens if t.lower() in [w.lower() for w in gen_tokens]]
    ai_percentage = len(ai_tokens) / len(text_tokens) * 100
    return ai_percentage

# Get input text from user
text = input("Enter some text to check for AI content: ")

# Call the detect_ai_text function
ai_percentage = detect_ai_text(text)

# Print the result
print(f"The likelihood that this text contains AI-related content is {ai_percentage:.2f}%")
