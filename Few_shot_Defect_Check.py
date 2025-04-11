import os
import google.generativeai as genai
from PIL import Image  # For image handling
import io
import time
import pandas as pd
import base64
import traceback # Import traceback for detailed error info
import json

import tiktoken

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
  """Counts the number of tokens in a string using a specified OpenAI model's tokenizer.

  Args:
      text: The input string.
      model_name: The name of the OpenAI model to use for tokenization.
                    Defaults to "gpt-3.5-turbo".

  Returns:
      The number of tokens in the string.  Returns -1 if the encoding is not found.
  """
  try:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens
  except KeyError:
    print(f"Warning: Model name '{model_name}' not recognized.  Falling back to cl100k_base. Consider updating your tiktoken library.")
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback to a common encoding
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except:
        print("Error: cl100k_base encoding not found. Please ensure tiktoken is properly installed.")
        return -1 # Indicate encoding not found

  except Exception as e:
    print(f"An error occurred during tokenization: {e}")
    return -1 # Indicate an error occurred
  
#Replace with your actual Gemini API key (securely stored)
GOOGLE_API_KEY = ""  # DO NOT HARDCODE IN REAL APPLICATIONS
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash') #for images models is pro-vision, for normal text models is only pro


def initialize_gemini():
    """Initializes the Gemini API (replace with actual API setup)."""

    # Load the Gemini Pro Vision model. You will likely want to initialize this globally for performance reasons.
    # Replace with how you initialize your model within the Gemini API.

    return model


def encode_image(image_path):
    """Encodes an image to base64 with better error handling."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        traceback.print_exc()  # Print full traceback
        return None

def create_few_shot_prompt(good_examples, scratch_examples, liquid_examples, hole_examples):
    """Creates a few-shot prompt for the Gemini model.  Handles potential encoding errors."""

    examples = []

    def add_examples(img_paths, label):
        for img_path in img_paths:
            base64_image = encode_image(img_path)
            if base64_image:
                examples.append({"image": base64_image, "label": label})
            else:
                print(f"Skipping {img_path} due to encoding error.")  # Log skipped images

    add_examples(good_examples, "no defect")
    #add_examples(scratch_examples, "defect, scratch")
    #add_examples(liquid_examples, "defect, liquid")
    #add_examples(hole_examples, "defect, hole")

    return examples
def analyze_image(image_path, gemini_model, few_shot_examples):
    """Analyzes an image using the Gemini API with few-shot prompting.  Robust error handling."""

    start_time = time.time()
    defect_status = "unknown"
    defect_type = None
    try:
        #img = Image.open(image_path)  # Not used, remove
        base64_image = encode_image(image_path)  # Encode the input image

        if not base64_image:
            print(f"Failed to encode input image {image_path}")
            return "error", None, time.time() - start_time  # Return immediately on error

        contents = []
        print("7a")

        # Add few-shot examples to the request
        for example in few_shot_examples:
            contents.append({"mime_type": "image/png", "data": example["image"]})  # Assuming JPEG
            contents.append({"text": example["label"]})
        print("7b")
        # Add the input image and prompt
        contents.append({"mime_type": "image/png", "data": base64_image})  # send as image with the list of files and data for classification
        contents.append({"text": "Analyze the image for defects. Return 'defect' or 'no defect'. If defect, indicate 'scratch', 'liquid', or 'hole'."})
        print("7c")
        try:
            
            #print(count_tokens(contents))
            with open("output.txt", "w") as file:
                json.dump(contents, file, indent=4)
            response = gemini_model.generate_content(contents)  # send to gemini
            response.resolve()
            
            print(response)
            response_text = response.text.lower()
            print("7d")

            if "no defect" in response_text:
                defect_status = "no defect"
                defect_type = None
            elif "defect" in response_text:
                defect_status = "defect"
                if "scratch" in response_text:
                    defect_type = "scratch"
                elif "liquid" in response_text:
                    defect_type = "liquid"
                elif "hole" in response_text:
                    defect_type = "hole"
                else:
                    defect_type = "unknown"

            else:
                defect_status = "uncertain" # Add case for when the model doesn't provide a clear answer
                defect_type = None
                print(f"Unclear response for {image_path}: {response_text}") # Log unclear answers

            end_time = time.time()
            request_time = end_time - start_time
            print(f"Image analyzed at: {image_path} --- Raw Response: {response_text} -- Request Time: {request_time:.4f} seconds")


        except Exception as e:
            end_time = time.time()
            request_time = end_time - start_time
            print(f"Gemini API error analyzing image {image_path}: {e} -- Request time: {request_time:.4f} seconds")
            traceback.print_exc()  # Print the traceback
            defect_status = "api_error"  # Distinct error status
            defect_type = None



    except Exception as e:
        end_time = time.time()
        request_time = end_time - start_time
        print(f"Error analyzing image {image_path}: {e} -- Request time: {request_time:.4f} seconds")
        traceback.print_exc() # Capture and display the traceback
        defect_status = "error"
        defect_type = None
    return defect_status, defect_type, request_time


def process_folder(folder_path, gemini_model, model_name, few_shot_examples):
    """Processes all images in a folder, adding timing and recording results.  Skips files on error."""
    results = []
    print("p6")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print("p7")
            defect_status, defect_type, request_time = analyze_image(image_path, gemini_model, few_shot_examples)
            print("p8")
            results.append({
                'filename': filename,
                'model_name': model_name,
                'Actual_defect_type': os.path.basename(folder_path),
                'Predicted_defect_status': defect_status,
                'Predicted_defect_type': defect_type,
                'request_time': request_time
            })

            print(f"{filename}: Defect Status = {defect_status}, Defect Type = {defect_type}, Request Time = {request_time:.4f}")
        else:
            print(f"Skipping non-image file: {filename}")

    return results



def main():
    """Main function to orchestrate the image analysis."""
    # Initialize Gemini
    gemini_model = initialize_gemini()

    if gemini_model is None:
        print("Failed to initialize Gemini. Exiting.")
        return  # Exit if initialization failed

    model_name = 'gemini-pro-vision' # Make sure this is the correct model name

    # Define folders
    good_folder = #folder link
    scratch_folder = #folder link
    liquid_folder = #folder link
    hole_folder = #folder link
    print("p1")
    # Select a few example images for each class (adjust the number as needed)
    num_examples = 1
    def get_image_examples(folder, num_examples):
        return [os.path.join(folder, f) for f in os.listdir(folder)[:num_examples] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print("p2")
    good_examples = get_image_examples(good_folder, num_examples)
    scratch_examples = get_image_examples(scratch_folder, num_examples)
    liquid_examples = get_image_examples(liquid_folder, num_examples)
    hole_examples = get_image_examples(hole_folder, num_examples)
    print("p3")

    # Create the few-shot examples
    few_shot_examples = create_few_shot_prompt(good_examples, scratch_examples, liquid_examples, hole_examples)
    print("p4")

    # Process defect folders using the few-shot examples
    hole_results = process_folder(hole_folder, gemini_model, model_name, few_shot_examples)
    good_results = process_folder(good_folder, gemini_model, model_name, few_shot_examples)
    scratch_results = process_folder(scratch_folder, gemini_model, model_name, few_shot_examples)
    liquid_results = process_folder(liquid_folder, gemini_model, model_name, few_shot_examples)
    print("p5")
    # Combine results into a single list
    all_results = hole_results + good_results + scratch_results + liquid_results

    # Create a Pandas DataFrame from the combined results:
    df_results = pd.DataFrame(all_results)
    # Reorder the columns in a specific and understandable way (customize as needed)
    column_order = ['filename', 'model_name', 'Actual_defect_type', 'Predicted_defect_status',
                    'Predicted_defect_type',
                    'request_time']  # you might need to import model confidence in classification and also add in report

    df_results = df_results[column_order]  # order list with correct fields

    # Print (or save) the results table
    print("\nResults Table:")
    print(df_results)

    # Optionally saving with date stamp

    results_filename = "gemini_image_analysis_results.csv"  # basic default values filename name and formats and extension

    if os.path.exists(results_filename):  # basic option will keep name the old report and create a datatime file name new
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name, extension = os.path.splitext(results_filename)  # splitting parts and creating basic new report as YYYYMMDD_HHMMSS . extension "gemini_image_analysis_results" / "CSV"
        results_filename = f"{base_name}_{timestamp}{extension}"  # create "gemini_image_analysis_results_YYYYMMDD_HHMMSS.csv"

    df_results.to_csv(results_filename, index=False)
    print(f"\nResults saved to: {results_filename}")


if __name__ == "__main__":
    main()
