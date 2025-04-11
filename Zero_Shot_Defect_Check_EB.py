import os
import google.generativeai as genai
#from google.generativeai.types import Part
from PIL import Image  # For image handling (install with `pip install Pillow`)
import io
import time
import pandas as pd

#Replace with your actual Gemini API key (securely stored)
GOOGLE_API_KEY = ""  # DO NOT HARDCODE IN REAL APPLICATIONS
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash') #for images models is pro-vision, for normal text models is only pro

# Constants for handling rate limiting
MAX_RETRIES = 3  # Maximum number of retries
INITIAL_DELAY = 5  # Initial delay in seconds before retrying
MAX_DELAY = 60  # Maximum delay in seconds


def initialize_gemini():
    """Initializes the Gemini API (replace with actual API setup)."""

    # Load the Gemini Pro Vision model. You will likely want to initialize this globally for performance reasons.
    # Replace with how you initialize your model within the Gemini API.

    return model


def analyze_image(image_path, gemini_model):
    """Analyzes an image using the Gemini API for object detection and defect type.  Includes timing.

    Args:
        image_path: Path to the image file.
        gemini_model: Initialized Gemini model.

    Returns:
        A tuple containing:
        - 'defect' or 'no defect' (string)
        - The type of defect ('scratch', 'liquid', 'hole', or None if no defect)
        - The time taken for the API request (in seconds)
    """
    start_time = time.time()
    defect_status = "unknown" # initialize these vars
    defect_type = None
    retries = 0
    delay = INITIAL_DELAY

    while retries < MAX_RETRIES:
        try:
            # 1. Load the image:
            img = Image.open(image_path)

            # 2. Construct the Gemini API request:
            # ADAPT TO YOUR API NEEDS
            prompt_text = "Analyze the image for defects (scratches, liquid, holes). Return 'defect' or 'no defect'. If defect present indicate 'scratch','liquid', or 'hole'."

            response = gemini_model.generate_content([prompt_text, img])
            response.resolve()

            end_time = time.time()
            request_time = end_time - start_time
            print(f"Image analyzed at: {image_path} --- Raw Response: {response.text} -- Request Time: {request_time:.4f} seconds")


            # 3. Parse the API response:
            response_text = response.text.lower()


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
                    defect_type = "unknown" # you should change this with better results on Prompt

            return defect_status, defect_type, request_time  # Success! Return the results.

        except Exception as e:
            end_time = time.time() #try get at least execution
            request_time = end_time - start_time
            print(f"Error analyzing image {image_path}: {e} -- Request time: {request_time:.4f} seconds")

            if "429 Resource has been exhausted" in str(e):  # Check for the rate limit error explicitly
                retries += 1
                print(f"Rate limit hit. Retrying in {delay} seconds (attempt {retries}/{MAX_RETRIES}).")
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)  # Exponential backoff
            else:
                # Other errors are not retried
                print(f"Non-retryable error analyzing image {image_path}: {e}")
                defect_status = "error"
                defect_type = None
                return defect_status, defect_type, request_time

    # If we reach here, it means we retried MAX_RETRIES times and failed
    print(f"Failed to analyze image {image_path} after {MAX_RETRIES} retries.")
    defect_status = "error"
    defect_type = None
    return defect_status, defect_type, request_time




def process_folder(folder_path, gemini_model, model_name):
    """Processes all images in a folder, adding timing and recording results.

    Args:
        folder_path: Path to the folder containing images.
        gemini_model: Initialized Gemini model.
        model_name: The name of the model being used (e.g., 'gemini-pro-vision').

    Returns:
        A list of dictionaries, where each dictionary represents the analysis
        result for one image.
    """
    results = []  # Store results (list of dictionaries)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            image_path = os.path.join(folder_path, filename)
            defect_status, defect_type, request_time = analyze_image(image_path, gemini_model)

            results.append({
                'filename': filename,
                'model_name': model_name,
                'Actual_defect_type': os.path.basename(folder_path),
                'Predicted_defect_status': defect_status,
                'Predicted_defect_type': defect_type,
                'request_time': request_time
            })


            print(f"{filename}: Defect Status = {defect_status}, Defect Type = {defect_type}, Request Time = {request_time:.4f}")


    return results

def main():
    """Main function to orchestrate the image analysis."""
    # Initialize Gemini
    gemini_model = initialize_gemini()
    model_name = 'gemini-1.5-flash'
    # Define folders
    good_folder = 'C:/Users/chika/Downloads/Wood stuff/no defect'
    scratch_folder = "C:/Users/chika/Downloads/Wood stuff/scratch"
    liquid_folder = "C:/Users/chika/Downloads/Wood stuff/liquid"
    hole_folder = "C:/Users/chika/Downloads/Wood stuff/hole"


   
    # Process defect folders and record the values that has been created, or it does exists the file will
    hole_results = process_folder(hole_folder, gemini_model, model_name)
    good_results = process_folder(good_folder, gemini_model, model_name) 
    scratch_results = process_folder(scratch_folder, gemini_model, model_name)
    liquid_results = process_folder(liquid_folder, gemini_model, model_name)
     #Analyze folder for errors on Good Pics, useful for improving detection on Defect AI and classifying errors that AI do

    # Combine results into a single list
    all_results =  hole_results+good_results +good_results + scratch_results+liquid_results

    # Create a Pandas DataFrame from the combined results:
    df_results = pd.DataFrame(all_results)
    # Reorder the columns in a specific and understandable way (customize as needed)
    column_order = ['filename', 'model_name','Actual_defect_type','Predicted_defect_status', 'Predicted_defect_type', 'request_time'] # you might need to import model confidence in classification and also add in report

    df_results = df_results[column_order] #order list with correct fields

    # Print (or save) the results table
    print("\nResults Table:")
    print(df_results)

    #Optionally saving with date stamp

    results_filename = "gemini_image_analysis_results.csv" #basic default values filename name and formats and extension

    if os.path.exists(results_filename): #basic option will keep name the old report and create a datatime file name new
         timestamp = time.strftime("%Y%m%d_%H%M%S")
         base_name, extension = os.path.splitext(results_filename) #splitting parts and creating basic new report as YYYYMMDD_HHMMSS . extension example "gemini_image_analysis_results" / "CSV"
         results_filename = f"{base_name}_{timestamp}{extension}" # create "gemini_image_analysis_results_YYYYMMDD_HHMMSS.csv"

    df_results.to_csv(results_filename, index=False)
    print(f"\nResults saved to: {results_filename}")
if __name__ == "__main__":
    main()